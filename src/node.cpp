#include "QuadUkf.h"

int main(int argc, char **argv) {
  ros::init(argc, argv, "kalman_sense");
  ros::NodeHandle nh;

  ros::Publisher poseStampedPub = nh.advertise<geometry_msgs::PoseStamped>(
      "pose", 1);
  ros::Publisher poseWithCovStampedPub = nh
      .advertise<geometry_msgs::PoseWithCovarianceStamped>("poseWithCov", 1);
  ros::Publisher poseArrayPub = nh.advertise<geometry_msgs::PoseArray>(
      "poseHistory", 1);

  QuadUkf ukf = QuadUkf(poseStampedPub, poseWithCovStampedPub, poseArrayPub);

  ros::Subscriber imu_sub = nh.subscribe("/imu/data_raw", 1,
                                         &QuadUkf::imuCallback, &ukf);
  ros::Subscriber pose_sub = nh.subscribe("/vslam/pose", 1,
                                          &QuadUkf::poseCallback, &ukf);
  ros::spin();
  return 0;
}
