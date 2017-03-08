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

class QuadUkf : public UnscentedKf {
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
  struct QuadState {
    Eigen::Vector3d position;
    Eigen::Quaterniond quaternion;
    Eigen::Vector3d velocity;
    Eigen::Vector3d angular_velocity;
    Eigen::Vector3d acceleration;
  };

  struct QuadBelief {
    double timeStamp;
    double dt;
    QuadUkf::QuadState state;
    Eigen::MatrixXd covariance;
  } lastBelief;

  geometry_msgs::PoseWithCovarianceStamped lastPoseMsg;
  geometry_msgs::PoseArray quadPoseArray;
  const int POSE_ARRAY_SIZE = 10000;

  const Eigen::Vector3d GRAVITY_ACCEL { 0, 0, -9.81 };

  std::timed_mutex mtx;

  Eigen::MatrixXd Q_ProcNoiseCov, R_SensorNoiseCov, H_SensorMap;

  ros::Publisher poseStampedPublisher;
  ros::Publisher poseWithCovStampedPublisher;
  ros::Publisher poseArrayPublisher;

  geometry_msgs::PoseStamped quadBeliefToPoseStamped(
      const QuadUkf::QuadBelief b) const;
  geometry_msgs::PoseWithCovarianceStamped quadBeliefToPoseWithCovStamped(
      const QuadUkf::QuadBelief b) const;
  void publishAllPoseMessages(QuadUkf::QuadBelief b);
  void updatePoseArray(const geometry_msgs::PoseWithCovarianceStamped p);

  Eigen::Quaterniond checkQuat(const Eigen::Quaterniond lastQuat,
                               const Eigen::Quaterniond nextQuat) const;
  Eigen::MatrixXd generateBigOmegaMat(
      const Eigen::Vector3d angular_velocity) const;
  Eigen::VectorXd quadStateToEigen(const QuadUkf::QuadState qs) const;
  QuadUkf::QuadState eigenToQuadState(const Eigen::VectorXd x) const;
};

#endif  // QUADUKF_H_

