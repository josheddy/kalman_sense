#include "QuadUkf.h"
#include <iostream>

QuadUkf::QuadUkf(ros::Publisher pub)
{
  std::cout << "ctor started" << std::endl;

  publisher = pub;
  gravityAcc << 0.0, 0.0, -9.81;

  // Set up mean weights and covariance weights
  meanWeights = Eigen::VectorXd::Zero(2 * numStates + 1);
  meanWeights(0) = lambda / (numStates + lambda);
  for (int i = 1; i < numStates; i++)
  {
    meanWeights(i) = 1 / (2 * numStates + lambda);
  }
  covarianceWeights = meanWeights;
  covarianceWeights(0) += (1 - alpha * alpha + beta);

  // Define initial position, orientation, velocity, angular velocity, and acceleration
  Eigen::Quaterniond initQuat = Eigen::Quaterniond::Identity();
  Eigen::Vector3d initPosition, initVelocity, initAngVel, initAcceleration;
  initPosition = Eigen::Vector3d::Zero();
  initVelocity = Eigen::Vector3d::Zero();
  initAngVel = Eigen::Vector3d::Zero();
  initAcceleration = Eigen::Vector3d::Zero();

  // Define initial belief
  QuadUkf::QuadState initState {initPosition, initQuat, initVelocity,
                                initAngVel, initAcceleration};
  Eigen::MatrixXd initCov = Eigen::MatrixXd::Identity(numStates, numStates);
  initCov = initCov * 0.01;
  double initTimeStamp = ros::Time::now().toSec();
  double init_dt = 0.0001;
  QuadUkf::QuadBelief initBelief {initTimeStamp, init_dt, initState, initCov};
  lastBelief = initBelief;

  // Initialize process noise covariance and sensor noise covariance
  Q_ProcNoiseCov = Eigen::MatrixXd::Identity(numStates, numStates);
  Q_ProcNoiseCov *= 0.01;  // default value
  R_SensorNoiseCov = Eigen::MatrixXd::Identity(7, 7);
  R_SensorNoiseCov *= 0.01;  // default value

  // Linear sensor map matrix H (z = H * x)
  H_SensorMap = Eigen::MatrixXd::Identity(numStates, numStates);
  H_SensorMap.block<6, 6>(7, 7) = Eigen::MatrixXd::Zero(6, 6);

  std::cout << quadStateToEigen(lastBelief.state).transpose() << std::endl;
  std::cout << "ctor finished" << std::endl;
}

QuadUkf::~QuadUkf()
{
}

/*
 * Predicts next state based on IMU readings, resets lastBelief to
 * reflect that prediction, then publishes lastBelief as a
 * geometry_msgs::PoseWithCovarianceStamped message.
 */
void QuadUkf::imuCallback(const sensor_msgs::ImuConstPtr &msg_in)
{
  std::cout << "imu cb started" << std::endl;

  // Compute time step "dt"
  double now = ros::Time::now().toSec();
  lastBelief.dt = now - msg_in->header.stamp.toSec();

  // Extract inertial data from IMU message
  lastBelief.state.angular_velocity(0) = msg_in->angular_velocity.x;
  lastBelief.state.angular_velocity(1) = msg_in->angular_velocity.y;
  lastBelief.state.angular_velocity(2) = msg_in->angular_velocity.z;
  lastBelief.state.acceleration(0) = msg_in->linear_acceleration.x;
  lastBelief.state.acceleration(1) = msg_in->linear_acceleration.y;
  lastBelief.state.acceleration(2) = msg_in->linear_acceleration.z;

  std::cout << "IMU data read in" << std::endl;
  std::cout << quadStateToEigen(lastBelief.state) << std::endl;

  // Predict next state and reset lastBelief
  Eigen::VectorXd x = quadStateToEigen(lastBelief.state);
  UnscentedKf::Belief b = predictState(x, lastBelief.covariance, Q_ProcNoiseCov,
                                       lastBelief.dt);
  QuadUkf::QuadBelief qb {now, lastBelief.dt, eigenToQuadState(b.state),
                          b.covariance};
  lastBelief = qb;

  std::cout << "post-prediction:" << std::endl;
  std::cout << quadStateToEigen(lastBelief.state) << std::endl;

  // Publish new pose message
  geometry_msgs::PoseWithCovarianceStamped msg_out;
  msg_out = quadBeliefToPoseWithCovStamped(lastBelief);
  publisher.publish(msg_out);
  std::cout << "imu cb finished" << std::endl;
}

/*
 * Corrects state based on pose sensor readings (that is, SLAM) and resets
 * lastBelief. Then it publishes lastBelief.
 */
void QuadUkf::poseCallback(
    const geometry_msgs::PoseWithCovarianceStampedConstPtr &msg_in)
{
  // Determine time step "dt" and set time stamp
  double now = ros::Time::now().toSec();
  lastBelief.dt = now - msg_in->header.stamp.toSec();
  lastBelief.timeStamp = now;

  // Extract pose information from pose sensor message
  Eigen::VectorXd z = Eigen::VectorXd::Zero(16);
  z(0) = msg_in->pose.pose.position.x;
  z(1) = msg_in->pose.pose.position.y;
  z(2) = msg_in->pose.pose.position.z;
  z(3) = msg_in->pose.pose.orientation.w;
  z(4) = msg_in->pose.pose.orientation.x;
  z(5) = msg_in->pose.pose.orientation.y;
  z(6) = msg_in->pose.pose.orientation.z;

  // Correct belief and reset lastBelief
  Eigen::VectorXd x = quadStateToEigen(lastBelief.state);
  Eigen::MatrixXd P = lastBelief.covariance;
  UnscentedKf::Belief currStateAndCov = correctState(x, P, z, R_SensorNoiseCov);
  lastBelief.state = eigenToQuadState(currStateAndCov.state);
  lastBelief.covariance = currStateAndCov.covariance;

  // Publish new pose message
  geometry_msgs::PoseWithCovarianceStamped msg_out;
  msg_out = quadBeliefToPoseWithCovStamped(lastBelief);
  publisher.publish(msg_out);
}

geometry_msgs::PoseWithCovarianceStamped QuadUkf::quadBeliefToPoseWithCovStamped(
    QuadUkf::QuadBelief b)
{
  geometry_msgs::PoseWithCovarianceStamped p;
  p.header.stamp.sec = b.timeStamp;
  p.header.stamp.nsec = (b.timeStamp - floor(b.timeStamp)) * pow(10, 9);
  p.pose.pose.position.x = b.state.position(0);
  p.pose.pose.position.y = b.state.position(1);
  p.pose.pose.position.z = b.state.position(2);
  p.pose.pose.orientation.w = b.state.quaternion.w();
  p.pose.pose.orientation.x = b.state.quaternion.x();
  p.pose.pose.orientation.y = b.state.quaternion.y();
  p.pose.pose.orientation.z = b.state.quaternion.z();

  //TODO Figure out covariance translation from QuadBelief to 6-by-6 PwCS representation
  // Copy covariance matrix from b into the covariance array in p
//  int numCovElems = b.covariance.rows() * b.covariance.cols();
//  for (int i = 0; i < numCovElems; ++i)
//  {
//    p.pose.covariance[i] = b.covariance(i);
//  }

  return p;
}

Eigen::VectorXd QuadUkf::processFunc(const Eigen::VectorXd x, const double dt)
{
  QuadUkf::QuadState prevState = eigenToQuadState(x);
  QuadUkf::QuadState currState;

  // Compute orientation
  Eigen::Quaterniond gyroQuat;
  gyroQuat.coeffs() = 0, prevState.angular_velocity(0), prevState.angular_velocity(1), prevState.angular_velocity(2);
  Eigen::MatrixXd Omega = generateBigOmegaMat(prevState.angular_velocity);
  //currState.quaternion.coeffs() = prevState.quaternion.coeffs()
  //    + 0.5 * Omega * prevState.quaternion.coeffs() * dt;
  currState.quaternion.coeffs() = prevState.quaternion.coeffs()
        + 0.5 * Omega * gyroQuat.normalized() * dt;
  currState.quaternion.normalize();

  // Rotate current and previous accelerations into inertial frame, then average them
  Eigen::Vector3d inertialAcc;
  inertialAcc = (currState.quaternion.toRotationMatrix()
      * currState.acceleration
      + prevState.quaternion.toRotationMatrix() * prevState.acceleration) / 2.0;

  currState.velocity = prevState.velocity + (inertialAcc - gravityAcc) * dt;

  currState.position = prevState.position
      + ((currState.velocity + prevState.velocity) / 2.0) * dt;

  return quadStateToEigen(currState);
}

Eigen::VectorXd QuadUkf::observationFunc(const Eigen::VectorXd stateVec)
{
  return H_SensorMap * stateVec; // Currently assumes a linear observation model
}

/*
 * Returns the 4-by-4 Big Omega matrix for performing quaternion integration,
 * given a vector "w" of angular velocities in radians per second.
 */
Eigen::MatrixXd QuadUkf::generateBigOmegaMat(const Eigen::Vector3d w) const
{
  Eigen::MatrixXd Omega(4, 4);

  // Upper left 3-by-3 block: negative skew-symmetric matrix of vector w
  Omega(0, 0) = 0;
  Omega(0, 1) = w(2);
  Omega(0, 2) = -w(1);

  Omega(1, 0) = -w(2);
  Omega(1, 1) = 0;
  Omega(1, 2) = w(0);

  Omega(2, 0) = w(1);
  Omega(2, 1) = -w(0);
  Omega(2, 2) = 0;

  // Bottom left 1-by-3 block: negative transpose of vector w
  Omega.block<1, 3>(3, 0) = -w.transpose();

  // Upper right 3-by-1 block: w
  Omega.block<3, 1>(0, 3) = w;

  // Bottom right 1-by-1 block: 0
  Omega(3, 3) = 0;

  return Omega;
}

/*
 * Convert a QuadUkf::QuadState to an Eigen::VectorXd.
 */
Eigen::VectorXd QuadUkf::quadStateToEigen(const QuadUkf::QuadState qs) const
{
  Eigen::VectorXd x(numStates);
  x(0) = qs.position(0);
  x(1) = qs.position(1);
  x(2) = qs.position(2);

  x(3) = qs.quaternion.w();
  x(4) = qs.quaternion.x();
  x(5) = qs.quaternion.y();
  x(6) = qs.quaternion.z();

  x(7) = qs.velocity(0);
  x(8) = qs.velocity(1);
  x(9) = qs.velocity(2);

  x(10) = qs.angular_velocity(0);
  x(11) = qs.angular_velocity(1);
  x(12) = qs.angular_velocity(2);

  x(13) = qs.acceleration(0);
  x(14) = qs.acceleration(1);
  x(15) = qs.acceleration(2);

  return x;
}

/*
 * Convert an Eigen::VectorXd to a QuadUkf::QuadState.
 */
QuadUkf::QuadState QuadUkf::eigenToQuadState(const Eigen::VectorXd x) const
{
  QuadUkf::QuadState qs;
  qs.position(0) = x(0);
  qs.position(1) = x(1);
  qs.position(2) = x(2);

  qs.quaternion.w() = x(3);
  qs.quaternion.x() = x(4);
  qs.quaternion.y() = x(5);
  qs.quaternion.z() = x(6);

  qs.velocity(0) = x(7);
  qs.velocity(1) = x(8);
  qs.velocity(2) = x(9);

  qs.angular_velocity(0) = x(10);
  qs.angular_velocity(1) = x(11);
  qs.angular_velocity(2) = x(12);

  qs.acceleration(0) = x(13);
  qs.acceleration(1) = x(14);
  qs.acceleration(2) = x(15);

  return qs;
}
;
