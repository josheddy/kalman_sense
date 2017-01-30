#include "QuadUkf.h"

int main(int argc, char **argv)
{
  ros::init(argc, argv, "kalman_sense");
  ros::NodeHandle nh;
  ros::Publisher pub = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>(
      "pose", 1);
  QuadUkf ukf = QuadUkf(pub);
  ros::Subscriber imu_sub = nh.subscribe("/imu/data_raw", 1,
                                         &QuadUkf::imuCallback, &ukf);
  ros::Subscriber pose_sub = nh.subscribe("/vslam/pose", 1,
                                          &QuadUkf::poseCallback, &ukf);
  ros::spin();
  return 0;
}
