#include "ros/ros.h"
#include "geometry_msgs/PoseWithCovarianceStamped.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <complex>

// algorithm from Crassidis paper

int count = 0;
const int N = 250;
const double weight = 1.0 / N;
Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(4, N);

void poseCallback(const geometry_msgs::PoseWithCovarianceStampedConstPtr &msg)
{
  // PTAM's convention: (w, z, y, x)
  Q(0, count) = msg->pose.pose.orientation.x;
  Q(1, count) = msg->pose.pose.orientation.w;
  Q(2, count) = msg->pose.pose.orientation.z;
  Q(3, count) = msg->pose.pose.orientation.y;

  ++count;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "quat_average");
  ros::NodeHandle nh;
  ros::Subscriber sub = nh.subscribe("/vslam/pose", 1, &poseCallback);

  while (ros::ok() and count < N)
  {
    ros::spinOnce();
    std::cout << count << std::endl;
  }

  Eigen::MatrixXd M = Eigen::MatrixXd::Zero(4, 4);
  M = Q * Q.transpose();
  Eigen::EigenSolver<Eigen::MatrixXd> eigensolver(M);
  Eigen::VectorXcd lambdas = eigensolver.eigenvalues();
  Eigen::MatrixXcd vecs = eigensolver.eigenvectors();
  int maxIndex = 0;
  std::complex<double> tempVal = lambdas(0);
  double max = sqrt(pow(tempVal.real(), 2) + pow(tempVal.imag(), 2));
  for (int i = 1; i < lambdas.rows(); ++i)
  {
    tempVal = lambdas(i);
    double norm = sqrt(pow(tempVal.real(), 2) + pow(tempVal.imag(), 2));
    if (norm > max)
    {
      maxIndex = i;
    }
  }
  Eigen::VectorXcd maxVec = vecs.col(maxIndex);
  maxVec = maxVec.normalized();
  Eigen::Vector4d realVec = maxVec.real();
  std::cout << "The average quaternion is: \n" << realVec << std::endl;

  return 0;
}
