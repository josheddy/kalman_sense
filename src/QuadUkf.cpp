#include "QuadUkf.h"

QuadUkf::QuadUkf(ros::Publisher poseStampedPub,
                 ros::Publisher poseWithCovStampedPub,
                 ros::Publisher poseArrayPub)
{
  poseStampedPublisher = poseStampedPub;
  poseWithCovStampedPublisher = poseWithCovStampedPub;
  poseArrayPublisher = poseArrayPub;

  numStates = 16;
  numSensors = 10;
  this->UnscentedKf::setWeights();

  // Define initial position, quaternion, velocity, angular velocity, and
  // acceleration.
  Eigen::Quaterniond initQuat = Eigen::Quaterniond::Identity();
  Eigen::Vector3d initPosition, initVelocity, initAngVel, initAcceleration;
  initPosition << 0, 0, 1;  // "x = 0, y = 0, z = 1 meter above origin"
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

  // Initialize sensor noise covariance matrix
//  Eigen::MatrixXd mainBlock(7, 7);
//  mainBlock << 1.60733370, 0.65393149, 0.02947270, 0.00164179, 0.02865161, 0.00472339,-0.00554183,
//               0.65393149, 1.36437181, 0.02745309, 0.00088938, 0.01202295,-0.00082392,-0.00705118,
//               0.02947270, 0.02745309, 0.00100832, 0.00002936, 0.00049179, 0.00007580,-0.00020808,
//               0.00164179, 0.00088938, 0.00002936, 0.00000330, 0.00005820, 0.00000087,-0.00011233,
//               0.02865161, 0.01202295, 0.00049179, 0.00005820, 0.00106645, 0.00002199,-0.00011233,
//               0.00472339,-0.00082392, 0.00007580, 0.00000087, 0.00002199, 0.00003296,-0.00000482,
//              -0.00554183,-0.00705118,-0.00020808,-0.00000714,-0.00011233,-0.00000482, 0.00005254;
//  SensorCovMatrixR.block(0, 0, 7, 7) = mainBlock;
  SensorCovMatrixR = R_SCALING_COEFF
      * Eigen::MatrixXd::Identity(numSensors, numSensors);
  Q_ProcNoiseCov = Q_SCALING_COEFF
      * Eigen::MatrixXd::Identity(numStates, numStates);

  // Initialize last PTAM message for pseudovelocity corrections
  lastPoseMsg.header.stamp.sec = initTimeStamp;
  lastPoseMsg.pose.pose.position.x = initPosition(0);
  lastPoseMsg.pose.pose.position.y = initPosition(1);
  lastPoseMsg.pose.pose.position.z = initPosition(2);

  // Initialize pose array for visualization in Rviz
  quadPoseArray.poses.clear();
  quadPoseArray.header.frame_id = "map";
  quadPoseArray.header.stamp = ros::Time();
}

QuadUkf::QuadUkf(QuadUkf&& other)
{
  std::lock_guard<std::timed_mutex> lock(other.mtx);

  SensorCovMatrixR = std::move(other.SensorCovMatrixR);
  other.SensorCovMatrixR = Eigen::MatrixXd::Zero(1, 1);

  ros::NodeHandle n;
  poseStampedPublisher = std::move(other.poseStampedPublisher);
  poseWithCovStampedPublisher = std::move(other.poseWithCovStampedPublisher);
  poseArrayPublisher = std::move(other.poseArrayPublisher);
  ros::Publisher p = n.advertise<std_msgs::Empty>("empty", 1);
  other.poseWithCovStampedPublisher = p;
  other.poseArrayPublisher = p;
}

QuadUkf::~QuadUkf()
{
}

void QuadUkf::imuCallback(const sensor_msgs::ImuConstPtr &msg_in)
{
  mtx.try_lock_for(std::chrono::milliseconds(100));

  QuadBelief xHat = lastBelief;
  xHat.state.angular_velocity(0) = msg_in->angular_velocity.x;
  xHat.state.angular_velocity(1) = -msg_in->angular_velocity.y;
  xHat.state.angular_velocity(2) = msg_in->angular_velocity.z;
  xHat.state.acceleration(0) = -msg_in->linear_acceleration.x;
  xHat.state.acceleration(1) = msg_in->linear_acceleration.y;
  xHat.state.acceleration(2) = msg_in->linear_acceleration.z;

  // Remove gravity
  xHat.state.acceleration = xHat.state.acceleration
      - xHat.state.quaternion.toRotationMatrix().inverse() * GRAVITY_ACCEL;

  // Predict next state and reset lastBelief
  Eigen::VectorXd x = quadStateToEigen(xHat.state);
  xHat.dt = msg_in->header.stamp.toSec() - lastBelief.timeStamp;
  //Eigen::MatrixXd Q = (Q_SCALING_COEFF * ProcessCovMatrixQ(xHat.dt));
  Eigen::MatrixXd Q = Q_ProcNoiseCov;
  UnscentedKf::Belief b = predictState(x, xHat.covariance, Q, xHat.dt);
  QuadUkf::QuadBelief qb {msg_in->header.stamp.toSec(), xHat.dt,
                          eigenToQuadState(b.state), b.covariance};
  qb.state.quaternion = checkQuatContinuity(lastBelief.state.quaternion,
                                            qb.state.quaternion);
  lastBelief = qb;

  publishAllPoseMessages(lastBelief);

  mtx.unlock();
}

void QuadUkf::poseCallback(
    const geometry_msgs::PoseWithCovarianceStampedConstPtr &msg_in)
{
  mtx.try_lock_for(std::chrono::milliseconds(100));

  Eigen::VectorXd z(numSensors);
  z(POS_X) = -msg_in->pose.pose.position.x;
  z(POS_Y) = msg_in->pose.pose.position.y;
  z(POS_Z) = msg_in->pose.pose.position.z;
  z(QUAT_X) = msg_in->pose.pose.orientation.w;
  z(QUAT_Y) = -msg_in->pose.pose.orientation.z;
  z(QUAT_Z) = msg_in->pose.pose.orientation.y;
  z(QUAT_W) = msg_in->pose.pose.orientation.x;

  // Pseudovelocity correction
  double dtPose = msg_in->header.stamp.toSec()
      - lastPoseMsg.header.stamp.toSec();
  z(VEL_X) = (z(POS_X) - lastPoseMsg.pose.pose.position.x) / dtPose;
  z(VEL_Y) = (z(POS_Y) - lastPoseMsg.pose.pose.position.y) / dtPose;
  z(VEL_Z) = (z(POS_Z) - lastPoseMsg.pose.pose.position.z) / dtPose;

  // Update lastPoseMsg
  lastPoseMsg.header = msg_in->header;
  lastPoseMsg.pose.pose.position.x = z(POS_X);
  lastPoseMsg.pose.pose.position.y = z(POS_Y);
  lastPoseMsg.pose.pose.position.z = z(POS_Z);

  // Check incoming quaternion for rotational continuity and replace if not
  // continuous.
  Eigen::Quaterniond inQuat;
  inQuat.x() = z(QUAT_X);
  inQuat.y() = z(QUAT_Y);
  inQuat.z() = z(QUAT_Z);
  inQuat.w() = z(QUAT_W);
  Eigen::Vector4d chosenQuat = checkQuatContinuity(lastBelief.state.quaternion,
                                                   inQuat).coeffs();
  z.block<4, 1>(3, 0) = chosenQuat;

  // Set time step "dt".
  double dt = msg_in->header.stamp.toSec() - lastBelief.timeStamp;

  QuadUkf::QuadBelief xHat = lastBelief;
  xHat.state.velocity = lastBelief.state.velocity
      + lastBelief.state.acceleration * dt;
  xHat.state.position = (xHat.state.velocity + lastBelief.state.velocity) / 2.0
      * dt + lastBelief.state.position;
  Eigen::MatrixXd Theta = quatIntegrationMatrix(
      lastBelief.state.angular_velocity);
  xHat.state.quaternion.coeffs() = lastBelief.state.quaternion.coeffs()
      + 0.5 * Theta * lastBelief.state.quaternion.coeffs() * dt;
  xHat.state.quaternion.normalize();
  Eigen::VectorXd xPred = quadStateToEigen(xHat.state);
  Eigen::MatrixXd P = lastBelief.covariance;

  UnscentedKf::Belief currStateAndCov = correctState(xPred, P, z,
                                                     SensorCovMatrixR);

  // Update lastBelief.
  lastBelief.dt = dt;
  lastBelief.state = eigenToQuadState(currStateAndCov.state);
  lastBelief.covariance = currStateAndCov.covariance;
  lastBelief.timeStamp = msg_in->header.stamp.toSec();

  publishAllPoseMessages(lastBelief);

  mtx.unlock();
}

void QuadUkf::publishAllPoseMessages(const QuadUkf::QuadBelief b)
{
  const geometry_msgs::PoseWithCovarianceStamped pwcs =
      quadBeliefToPoseWithCovStamped(b);
  poseWithCovStampedPublisher.publish(pwcs);
  updatePoseArray(pwcs);
  const geometry_msgs::PoseStamped ps = quadBeliefToPoseStamped(b);
  poseStampedPublisher.publish(ps);
}

/*
 * Puts a given pose into the first position of quadPoseArray. Once
 * quadPoseArray reaches POSE_ARRAY_SIZE, the last pose is popped on each call.
 * After these operations are performed, this function publishes
 * quadPoseArray.
 */
void QuadUkf::updatePoseArray(const geometry_msgs::PoseWithCovarianceStamped p)
{
  quadPoseArray.poses.insert(quadPoseArray.poses.begin(), 1, p.pose.pose);
  if (quadPoseArray.poses.size() > POSE_ARRAY_SIZE)
  {
    quadPoseArray.poses.pop_back();
  }
  poseArrayPublisher.publish(quadPoseArray);
}

/*
 * Ensures rotational continuity by checking for sign-flipping in the
 * orientation quaternion.
 */
Eigen::Quaterniond QuadUkf::checkQuatContinuity(
    const Eigen::Quaterniond lastQuat, const Eigen::Quaterniond nextQuat) const
{
  Eigen::Vector4d lastVec, nextVec;
  lastVec = lastQuat.normalized().coeffs();
  nextVec = nextQuat.normalized().coeffs();

  double sum = (lastVec + nextVec).norm();
  double diff = (lastVec - nextVec).norm();

  Eigen::Quaterniond out;
  if (sum > diff)
  {
    out.coeffs() = nextVec;
  }
  else
  {
    out.coeffs() = -nextVec;
  }
  return out;
}

geometry_msgs::PoseStamped QuadUkf::quadBeliefToPoseStamped(
    const QuadUkf::QuadBelief b) const
{
  geometry_msgs::PoseStamped p;
  p.header.stamp.sec = b.timeStamp;
  p.header.stamp.nsec = (b.timeStamp - floor(b.timeStamp)) * pow(10, 9);
  p.pose.position.x = b.state.position(0);
  p.pose.position.y = b.state.position(1);
  p.pose.position.z = b.state.position(2);
  p.pose.orientation.w = b.state.quaternion.w();
  p.pose.orientation.x = b.state.quaternion.x();
  p.pose.orientation.y = b.state.quaternion.y();
  p.pose.orientation.z = b.state.quaternion.z();

  return p;
}

geometry_msgs::PoseWithCovarianceStamped QuadUkf::quadBeliefToPoseWithCovStamped(
    const QuadUkf::QuadBelief b) const
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
  Eigen::MatrixXd covMat = b.covariance.block<6, 6>(0, 0);
  for (int i = 0; i < covMat.rows() * covMat.cols(); ++i)
  {
    p.pose.covariance[i] = covMat(i);
  }

  return p;
}

Eigen::VectorXd QuadUkf::processFunc(const Eigen::VectorXd x, const double dt)
{
  QuadUkf::QuadState prevState = eigenToQuadState(x);
  prevState.quaternion.normalize();
  QuadUkf::QuadState currState;

  // Compute current orientation via quaternion integration.
  Eigen::MatrixXd Theta = quatIntegrationMatrix(prevState.angular_velocity);
  currState.quaternion.coeffs() = prevState.quaternion.coeffs()
      + 0.5 * Theta * prevState.quaternion.coeffs() * dt;
  currState.quaternion.normalize();

  // Rotate current and previous accelerations into inertial frame, then
  // average them.
  currState.acceleration = prevState.quaternion.toRotationMatrix()
      * prevState.acceleration;

  // Compute current velocity by integrating current acceleration.
  currState.velocity = prevState.velocity
      + 0.5 * (lastBelief.state.acceleration + currState.acceleration) * dt;

  // Compute current position by integrating current velocity.
  currState.position = prevState.position
      + ((currState.velocity + prevState.velocity) / 2.0) * dt;

  // Angular velocity is assumed to be correct as measured.
  currState.angular_velocity = prevState.angular_velocity;

  return quadStateToEigen(currState);
}

Eigen::VectorXd QuadUkf::observationFunc(const Eigen::VectorXd stateVec)
{
  return stateVec.head(numSensors);
}

/*
 * Given a vector of angular velocities in radians per second, returns the
 * 4-by-4 angular rate integration matrix.
 */
Eigen::MatrixXd QuadUkf::quatIntegrationMatrix(
    const Eigen::Vector3d angVel) const
{
  Eigen::MatrixXd Theta(4, 4);

  // Upper left 3-by-3 block: negative skew-symmetric matrix of vector w
  Theta(0, 0) = 0;
  Theta(0, 1) = angVel(2);
  Theta(0, 2) = -angVel(1);

  Theta(1, 0) = -angVel(2);
  Theta(1, 1) = 0;
  Theta(1, 2) = angVel(0);

  Theta(2, 0) = angVel(1);
  Theta(2, 1) = -angVel(0);
  Theta(2, 2) = 0;

  // Bottom left 1-by-3 block: negative transpose of vector w
  Theta.block<1, 3>(3, 0) = -angVel.transpose();

  // Upper right 3-by-1 block: w
  Theta.block<3, 1>(0, 3) = angVel;

  // Bottom right 1-by-1 block: 0
  Theta(3, 3) = 0;

  return Theta;
}

Eigen::VectorXd QuadUkf::quadStateToEigen(const QuadUkf::QuadState qs) const
{
  Eigen::VectorXd x(numStates);

  x(POS_X) = qs.position(0);
  x(POS_Y) = qs.position(1);
  x(POS_Z) = qs.position(2);

  x(QUAT_X) = qs.quaternion.x();
  x(QUAT_Y) = qs.quaternion.y();
  x(QUAT_Z) = qs.quaternion.z();
  x(QUAT_W) = qs.quaternion.w();

  x(VEL_X) = qs.velocity(0);
  x(VEL_Y) = qs.velocity(1);
  x(VEL_Z) = qs.velocity(2);

  x(ANGVEL_X) = qs.angular_velocity(0);
  x(ANGVEL_Y) = qs.angular_velocity(1);
  x(ANGVEL_Z) = qs.angular_velocity(2);

  x(ACCEL_X) = qs.acceleration(0);
  x(ACCEL_Y) = qs.acceleration(1);
  x(ACCEL_Z) = qs.acceleration(2);

  return x;
}

QuadUkf::QuadState QuadUkf::eigenToQuadState(const Eigen::VectorXd x) const
{
  QuadUkf::QuadState qs;

  qs.position(0) = x(POS_X);
  qs.position(1) = x(POS_Y);
  qs.position(2) = x(POS_Z);

  qs.quaternion.x() = x(QUAT_X);
  qs.quaternion.y() = x(QUAT_Y);
  qs.quaternion.z() = x(QUAT_Z);
  qs.quaternion.w() = x(QUAT_W);

  qs.velocity(0) = x(VEL_X);
  qs.velocity(1) = x(VEL_Y);
  qs.velocity(2) = x(VEL_Z);

  qs.angular_velocity(0) = x(ANGVEL_X);
  qs.angular_velocity(1) = x(ANGVEL_Y);
  qs.angular_velocity(2) = x(ANGVEL_Z);

  qs.acceleration(0) = x(ACCEL_X);
  qs.acceleration(1) = x(ACCEL_Y);
  qs.acceleration(2) = x(ACCEL_Z);

  return qs;
}

Eigen::MatrixXd QuadUkf::ProcessCovMatrixQ(const double dt) const
{
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(numStates, numStates);

  const double posVariance = IMU_ACCEL_STD_DEV * pow(dt, 5) / 20.0;
  Q.block(0, 0, 3, 3) = posVariance * Eigen::MatrixXd::Identity(3, 3);

  const double quatVariance = IMU_GYRO_STD_DEV * pow(dt, 3) / 3.0;
  Q.block(3, 3, 4, 4) = quatVariance * Eigen::MatrixXd::Identity(4, 4);

  const double velVariance = IMU_ACCEL_STD_DEV * pow(dt, 3) / 3.0;
  Q.block(7, 7, 3, 3) = velVariance * Eigen::MatrixXd::Identity(3, 3);

  const double gyroVariance = IMU_GYRO_STD_DEV * dt;
  Q.block(10, 10, 3, 3) = gyroVariance * Eigen::MatrixXd::Identity(3, 3);

  const double accelVariance = IMU_ACCEL_STD_DEV * dt;
  Q.block(13, 13, 3, 3) = accelVariance * Eigen::MatrixXd::Identity(3, 3);

  // Covariance (off-diagonal) terms //TODO
//  const double posVelCovariance = IMU_ACCEL_STD_DEV * pow(dt, 4) / 8.0;
//  Eigen::MatrixXd posVelBlock = posVelCovariance
//      * Eigen::MatrixXd::Identity(3, 3);
//  Q.block(0, 7, 3, 3) = posVelBlock;
//  Q.block(7, 0, 3, 3) = posVelBlock;
//
//  const double quatGyroCovariance = IMU_GYRO_STD_DEV * pow(dt, 2) / 2.0;
//  Eigen::MatrixXd quatGyroBlock = Eigen::MatrixXd::Constant(4, 3,
//                                                            quatGyroCovariance);
//  Q.block(3, 10, 4, 3) = quatGyroBlock;
//  Q.block(10, 3, 3, 4) = quatGyroBlock.transpose();
//
//  const double posAccelCovariance = IMU_ACCEL_STD_DEV * pow(dt, 3) / 6.0;
//  Eigen::MatrixXd posAccelBlock = posAccelCovariance
//      * Eigen::MatrixXd::Identity(3, 3);
//  Q.block(0, 13, 3, 3) = posAccelBlock;
//  Q.block(13, 0, 3, 3) = posAccelBlock;
//
//  const double velAccelCovariance = IMU_ACCEL_STD_DEV * pow(dt, 2) / 2.0;
//  Eigen::MatrixXd velAccelBlock = velAccelCovariance
//      * Eigen::MatrixXd::Identity(3, 3);
//  Q.block(7, 13, 3, 3) = velAccelBlock;
//  Q.block(13, 7, 3, 3) = velAccelBlock;

  return Q;
}
