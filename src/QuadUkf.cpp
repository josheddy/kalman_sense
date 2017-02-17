#include "QuadUkf.h"

QuadUkf::QuadUkf(ros::Publisher pub)
{
  publisher = pub;

  numStates = 16;
  numSensors = 7;
  this->UnscentedKf::setWeights();

  // Define initial position, orientation, velocity, angular velocity, and acceleration
  Eigen::Quaterniond initQuat = Eigen::Quaterniond::Identity();
  Eigen::Vector3d initPosition, initVelocity, initAngVel, initAcceleration;
  initPosition << 0, 0, 1;
  initVelocity = Eigen::Vector3d::Zero();
  initAngVel = Eigen::Vector3d::Zero();
  initAcceleration = Eigen::Vector3d::Zero();

  // Define initial belief
  QuadUkf::QuadState initState {initPosition, initQuat, initVelocity,
                                initAngVel, initAcceleration};
  Eigen::MatrixXd initCov = Eigen::MatrixXd::Identity(numStates, numStates);
  initCov = initCov * 0.01;
  //double initTimeStamp = ros::Time::now().toSec();
  now = ros::Time::now().toNSec();
  double init_dt = 0.0001;
  QuadUkf::QuadBelief initBelief {now, init_dt, initState, initCov};
  lastBelief = initBelief;

  // Initialize process noise covariance and sensor noise covariance
  Q_ProcNoiseCov = Eigen::MatrixXd::Identity(numStates, numStates);
  Q_ProcNoiseCov *= 0.01;  // default value
  R_SensorNoiseCov = Eigen::MatrixXd::Identity(numStates, numStates); //TODO add numMeasurements to UnscentedKf
  R_SensorNoiseCov *= 0.01;  // default value

  // Linear sensor map matrix H (z = H * x) //TODO create a more accurate sensor model?
  H_SensorMap = Eigen::MatrixXd::Zero(numStates, numSensors);
  H_SensorMap.block(0, 0, numSensors, numSensors) = Eigen::MatrixXd::Identity(
      numSensors, numSensors);
}

QuadUkf::QuadUkf(QuadUkf&& other)
{
  std::lock_guard<std::timed_mutex> lock(other.mtx);

  now = std::move(other.now);
  other.now = 0;

  Q_ProcNoiseCov = std::move(other.Q_ProcNoiseCov);
  other.Q_ProcNoiseCov = Eigen::MatrixXd::Zero(1, 1);

  R_SensorNoiseCov = std::move(other.R_SensorNoiseCov);
  other.R_SensorNoiseCov = Eigen::MatrixXd::Zero(1, 1);

  publisher = std::move(other.publisher);
  ros::NodeHandle n;
  ros::Publisher p = n.advertise<std_msgs::Empty>("empty", 1);
  other.publisher = p;

  H_SensorMap = std::move(other.H_SensorMap);
  other.H_SensorMap = Eigen::MatrixXd::Zero(1, 1);
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

  mtx.try_lock_for(std::chrono::milliseconds(100));

  QuadBelief xB = lastBelief;
  xB.state.angular_velocity(0) = msg_in->angular_velocity.x;
  xB.state.angular_velocity(1) = msg_in->angular_velocity.y;
  xB.state.angular_velocity(2) = msg_in->angular_velocity.z;
  xB.state.acceleration(0) = msg_in->linear_acceleration.x;
  xB.state.acceleration(1) = msg_in->linear_acceleration.y;
  xB.state.acceleration(2) = msg_in->linear_acceleration.z;

  // Remove gravity
  xB.state.acceleration = xB.state.acceleration
      - xB.state.quaternion.toRotationMatrix().inverse() * gravityAcc;

  std::cout << "IMU data read in" << std::endl;
  std::cout << quadStateToEigen(xB.state) << std::endl;

  // Predict next state and reset lastBelief
  Eigen::VectorXd x = quadStateToEigen(xB.state);
  now = ros::Time::now().toSec();
  xB.dt = now - msg_in->header.stamp.toSec();
  UnscentedKf::Belief b = predictState(x, xB.covariance, Q_ProcNoiseCov, xB.dt);
  QuadUkf::QuadBelief qb {now, xB.dt, eigenToQuadState(b.state), b.covariance};
  qb.state.quaternion.normalize();
  lastBelief = qb;

//  std::cout << "post-prediction:" << std::endl;
//  std::cout << std::fixed << std::setprecision(9)
//      << quadStateToEigen(lastBelief.state) << std::endl;

  // Publish new pose message
  geometry_msgs::PoseWithCovarianceStamped msg_out;
  msg_out = quadBeliefToPoseWithCovStamped(lastBelief);
  publisher.publish(msg_out);

  mtx.unlock();

  std::cout << "imu cb finished" << std::endl;
}

/*
 * Corrects state based on pose sensor readings (that is, SLAM) and resets
 * lastBelief. Then it publishes lastBelief.
 */
void QuadUkf::poseCallback(
    const geometry_msgs::PoseWithCovarianceStampedConstPtr &msg_in)
{
  std::cout << "pose cb started" << std::endl;

  mtx.try_lock_for(std::chrono::milliseconds(100));

  //now = ros::Time::now().toSec();
  // Extract pose information from pose sensor message
  Eigen::VectorXd z = Eigen::VectorXd::Zero(numStates);
  z(0) = msg_in->pose.pose.position.x;
  z(1) = msg_in->pose.pose.position.y;
  z(2) = msg_in->pose.pose.position.z;
  z(3) = msg_in->pose.pose.orientation.y; // PTAM's Quat Convention is backwards: (w, x, y, z)
  z(4) = msg_in->pose.pose.orientation.z;
  z(5) = msg_in->pose.pose.orientation.w;
  z(6) = msg_in->pose.pose.orientation.x;

  // Correct belief and reset lastBelief
  Eigen::VectorXd x = quadStateToEigen(lastBelief.state);
  Eigen::MatrixXd P = lastBelief.covariance;

  UnscentedKf::Belief currStateAndCov = correctState(x, P, z, R_SensorNoiseCov);

  //lastBelief.dt = now - msg_in->header.stamp.toSec();
  lastBelief.state = eigenToQuadState(currStateAndCov.state);
  lastBelief.state.quaternion.normalize();
  lastBelief.covariance = currStateAndCov.covariance;
  lastBelief.timeStamp = msg_in->header.stamp.toSec();

//  std::cout << "post-correction:" << std::endl;
//  std::cout << std::fixed << std::setprecision(9)
//      << quadStateToEigen(lastBelief.state) << std::endl;

  // Publish new pose message
  geometry_msgs::PoseWithCovarianceStamped msg_out;
  msg_out = quadBeliefToPoseWithCovStamped(lastBelief);
  publisher.publish(msg_out);

  mtx.unlock();

  std::cout << "pose cb finished" << std::endl;
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

  // Copy covariance matrix from b into the covariance array in p
  Eigen::MatrixXd relevant = b.covariance.block<6, 6>(0, 0);
  for (int i = 0; i < relevant.rows() * relevant.cols(); ++i)
  {
    p.pose.covariance[i] = relevant(i);
  }

  return p;
}

Eigen::VectorXd QuadUkf::processFunc(const Eigen::VectorXd x, const double dt)
{
  QuadUkf::QuadState prevState = eigenToQuadState(x);
  prevState.quaternion.normalize();
  QuadUkf::QuadState currState;

  // Compute current orientation via quaternion integration
  Eigen::MatrixXd Omega = generateBigOmegaMat(prevState.angular_velocity);
  currState.quaternion.coeffs() = prevState.quaternion.coeffs()
      + 0.5 * Omega * prevState.quaternion.coeffs() * dt;
  currState.quaternion.normalize();

  // Rotate current and previous accelerations into inertial frame, then average them
  currState.acceleration = prevState.quaternion.toRotationMatrix()
      * prevState.acceleration;

  // Sam's constant-jerk approximation:
  //Eigen::Vector3d prevA = prevState.quaternion.toRotationMatrix() * prevState.acceleration;
  //currState.acceleration =prevA + (prevA - lastBelief.state.acceleration)/(now-dt-lastBelief.timeStamp)*dt;

  // Compute current velocity by integrating current acceleration
  currState.velocity = prevState.velocity
      + 0.5 * (lastBelief.state.acceleration + currState.acceleration) * dt;

  // Compute current position by integrating current velocity
  currState.position = prevState.position
      + ((currState.velocity + prevState.velocity) / 2.0) * dt;

  // Angular velocity assumed to be correct as measured
  currState.angular_velocity = prevState.angular_velocity;

  return quadStateToEigen(currState);
}

Eigen::VectorXd QuadUkf::observationFunc(const Eigen::VectorXd sensorVec)
{
  return H_SensorMap * sensorVec; // Currently assumes a linear observation model
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

  x(3) = qs.quaternion.x();
  x(4) = qs.quaternion.y();
  x(5) = qs.quaternion.z();
  x(6) = qs.quaternion.w();

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

  qs.quaternion.x() = x(3);
  qs.quaternion.y() = x(4);
  qs.quaternion.z() = x(5);
  qs.quaternion.w() = x(6);

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
