// SimpleUKF.cpp

#include "SimpleUkf.h"

SimpleUkf::SimpleUkf(Eigen::VectorXd initStateVec, Eigen::VectorXd initSensorVec)
{
  numStates = initStateVec.rows();
  numSensors = initSensorVec.rows();

  meanWeights = Eigen::VectorXd::Zero(2 * numStates + 1);
  meanWeights(0) = lambda / (numStates + lambda);
  for (int i = 1; i < numStates; i++)
  {
    meanWeights(i) = 1 / (2 * numStates + lambda);
  }

  covarianceWeights = meanWeights;
  covarianceWeights(0) = covarianceWeights(0) + (1 - pow(alpha, 2) + beta);
}
;

Eigen::VectorXd SimpleUkf::processFunc(Eigen::VectorXd stateVec)
{
  Eigen::VectorXd x = stateVec;
  x(0) = stateVec(1);
  x(1) = stateVec(2);
  x(2) = 0.05 * stateVec(0) * (stateVec(1) + stateVec(2));
  return x;
}

Eigen::VectorXd SimpleUkf::observationFunc(Eigen::VectorXd stateVec)
{
  Eigen::VectorXd z(1);
  z << stateVec(0);
  return z;
}

SimpleUkf::~SimpleUkf()
{
}
