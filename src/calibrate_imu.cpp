#include "ros/ros.h"
#include "sensor_msgs/Imu.h"
#include "std_msgs/Empty.h"

double accBiasX = 0;
double accBiasY = 0;
double accBiasZ = 0;

//const double GRAVITY_ACCEL = -9.81;

int count = 0;

void collectBias(const sensor_msgs::ImuConstPtr &msg)
{
  accBiasX += msg->linear_acceleration.x;
  accBiasY += msg->linear_acceleration.y;
  accBiasZ += msg->linear_acceleration.z;
  ++count;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "kalman_sense");
  ros::NodeHandle nh;
  ros::Subscriber imu_sub = nh.subscribe("/imu/data_raw", 1, &collectBias);

  while (ros::ok() and count < 10000)
  {
    ros::spinOnce();
    std::cout << count << std::endl;
  }

  accBiasX = accBiasX / count;
  accBiasY = accBiasY / count;
  accBiasZ = accBiasZ / count;

  std::cout << "X, Y, and Z accelerometer biases:" << std::endl;
  std::cout << accBiasX << ", " << accBiasY << ", " << accBiasZ << std::endl;
  return 0;
}
