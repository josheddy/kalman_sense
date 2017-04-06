#ifndef QUADUKF_H_
#define QUADUKF_H_

#include "UnscentedKf.h"

#include <mutex>

#include "ros/ros.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/PoseWithCovarianceStamped.h"
#include "geometry_msgs/PoseArray.h"
#include "sensor_msgs/Imu.h"
#include "std_msgs/Empty.h"

class QuadUkf : public UnscentedKf
{
public:
  QuadUkf(ros::Publisher poseStampedPub, ros::Publisher poseWithCovStampedPub,
          ros::Publisher poseArrayPub);
  QuadUkf(QuadUkf&& other);
  ~QuadUkf();

  void imuCallback(const sensor_msgs::ImuConstPtr &msg_in);
  void poseCallback(
      const geometry_msgs::PoseWithCovarianceStampedConstPtr &msg_in);

  Eigen::VectorXd processFunc(const Eigen::VectorXd stateVec, const double dt);
  Eigen::VectorXd observationFunc(const Eigen::VectorXd stateVec);

private:
  struct QuadState
  {
    Eigen::Vector3d position;
    Eigen::Quaterniond quaternion;
    Eigen::Vector3d velocity;
    Eigen::Vector3d angular_velocity;
    Eigen::Vector3d acceleration;
  };

  struct QuadBelief
  {
    double timeStamp;
    double dt;
    QuadUkf::QuadState state;
    Eigen::MatrixXd covariance;
  } lastBelief;

  enum stateVars
  {
    POS_X = 0, POS_Y = 1, POS_Z = 2, QUAT_X = 3, QUAT_Y = 4, QUAT_Z = 5,
    QUAT_W = 6, VEL_X = 7, VEL_Y = 8, VEL_Z = 9, ANGVEL_X = 10, ANGVEL_Y = 11,
    ANGVEL_Z = 12, ACCEL_X = 13, ACCEL_Y = 14, ACCEL_Z = 15
  };

  Eigen::MatrixXd Q_ProcNoiseCov;
  Eigen::MatrixXd SensorCovMatrixR;
  const double IMU_ACCEL_STD_DEV = 8.66124974095918 * pow(10, -6);
  const double IMU_GYRO_STD_DEV = 1.2184696791468346 * pow(10, -7);
  const double Q_SCALING_COEFF = 0.01;
  const double R_SCALING_COEFF = 0.01;

  geometry_msgs::PoseWithCovarianceStamped lastPoseMsg;
  geometry_msgs::PoseArray quadPoseArray;
  const int POSE_ARRAY_SIZE = 10000;  // number of poses to keep for plotting

  const Eigen::Vector3d GRAVITY_ACCEL {0, 0, -9.81};

  std::timed_mutex mtx;

  Eigen::MatrixXd ProcessCovMatrixQ(const double dt) const;

  ros::Publisher poseStampedPublisher;
  ros::Publisher poseWithCovStampedPublisher;
  ros::Publisher poseArrayPublisher;

  geometry_msgs::PoseStamped quadBeliefToPoseStamped(const QuadBelief qb) const;
  geometry_msgs::PoseWithCovarianceStamped quadBeliefToPoseWithCovStamped(
      const QuadBelief qb) const;
  void publishAllPoseMessages(const QuadBelief qb);
  void updatePoseArray(const geometry_msgs::PoseWithCovarianceStamped p);

  Eigen::Quaterniond checkQuatContinuity(
      const Eigen::Quaterniond lastQuat,
      const Eigen::Quaterniond nextQuat) const;
  Eigen::MatrixXd quatIntegrationMatrix(const Eigen::Vector3d angVel) const;

  Eigen::VectorXd quadStateToEigen(const QuadUkf::QuadState qs) const;
  QuadUkf::QuadState eigenToQuadState(const Eigen::VectorXd x) const;
};

#endif  // QUADUKF_H_

