#ifndef QUADUKF_H_
#define QUADUKF_H_

#include "UnscentedKf.h"
#include "ros/ros.h"
#include "geometry_msgs/PoseWithCovarianceStamped.h"
#include "sensor_msgs/Imu.h"
#include <std_msgs/Empty.h>
#include <iostream>
#include <mutex>
#include <chrono>

class QuadUkf : public UnscentedKf
{
public:
  QuadUkf(ros::Publisher pub);
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

  double now;

  std::timed_mutex mtx;

  //const int numStates = 16; //TODO Do I still need this based on how numStates is set in UnscentedKf.cpp?
  Eigen::MatrixXd Q_ProcNoiseCov, R_SensorNoiseCov;

  ros::Publisher publisher;
  const Eigen::Vector3d GRAVITY_ACCEL{0, 0, -9.81}; // Gravity vector in inertial frame
  Eigen::MatrixXd H_SensorMap; // Observation model matrix H

  geometry_msgs::PoseWithCovarianceStamped quadBeliefToPoseWithCovStamped(
      QuadUkf::QuadBelief b);
  Eigen::MatrixXd generateBigOmegaMat(
      const Eigen::Vector3d angular_velocity) const;
  Eigen::VectorXd quadStateToEigen(const QuadUkf::QuadState qs) const;
  QuadUkf::QuadState eigenToQuadState(const Eigen::VectorXd x) const;
};

#endif /* QUADUKF_H_ */
