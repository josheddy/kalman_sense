#ifndef QUADUKF_H_
#define QUADUKF_H_

#include "UnscentedKf.h"
#include "ros/ros.h"
#include "geometry_msgs/PoseWithCovarianceStamped.h"
#include "sensor_msgs/Imu.h"
#include <iostream>

class QuadUkf : public UnscentedKf
{
public:
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

  //Transform lastStateTf; //TODO delete all instances of lastStateTf (deprecated)

  const int _numStates = 16;
  Eigen::MatrixXd Q_ProcNoiseCov, R_SensorNoiseCov;

  QuadUkf(ros::Publisher pub);
  ~QuadUkf();

  void imuCallback(const sensor_msgs::ImuConstPtr &msg_in);
  void poseCallback(
      const geometry_msgs::PoseWithCovarianceStampedConstPtr &msg_in);

  Eigen::VectorXd processFunc(const Eigen::VectorXd stateVec, const double dt);
  Eigen::VectorXd observationFunc(const Eigen::VectorXd stateVec);

private:
  ros::Publisher publisher;
  Eigen::Vector3d kGravityAcc; // Gravity vector in inertial frame
  Eigen::MatrixXd H_SensorMap; // Observation model matrix H

  geometry_msgs::PoseWithCovarianceStamped quadBeliefToPoseWithCovStamped(
      QuadUkf::QuadBelief b);
  Eigen::MatrixXd generateBigOmegaMat(
      const Eigen::Vector3d angular_velocity) const;
  Eigen::VectorXd quadStateToEigen(const QuadUkf::QuadState qs) const;
  QuadUkf::QuadState eigenToQuadState(const Eigen::VectorXd x) const;
};

#endif /* QUADUKF_H_ */
