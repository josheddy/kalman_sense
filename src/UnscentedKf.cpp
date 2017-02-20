#include "UnscentedKf.h"
#include <iostream>

UnscentedKf::UnscentedKf() :
    numStates(1), numSensors(1)
{
  this->setWeights();
}

UnscentedKf::~UnscentedKf()
{
}

UnscentedKf::Belief UnscentedKf::predictState(Eigen::VectorXd x,
                                              Eigen::MatrixXd P,
                                              Eigen::MatrixXd Q, double dt)
{
  // Compute sigma points around current estimated state
  Eigen::MatrixXd sigmaPts(numStates, 2 * numStates + 1);
  sigmaPts = computeSigmaPoints(x, P, SIGMA_PT_SCALING_COEFF);

  // Perform unscented transform on current estimated state to predict next
  // state and covariance
  UnscentedKf::Transform tf = unscentedStateTransform(sigmaPts, meanWeights,
                                                      covarianceWeights, Q, dt);

  // Return a new belief
  UnscentedKf::Belief bel {tf.vector, tf.covariance};
  return bel;
}

UnscentedKf::Belief UnscentedKf::correctState(Eigen::VectorXd x,
                                              Eigen::MatrixXd P,
                                              Eigen::VectorXd z,
                                              Eigen::MatrixXd R)
{
  Eigen::MatrixXd sigmaPts(numStates, 2 * numStates + 1);
  sigmaPts = computeSigmaPoints(x, P, SIGMA_PT_SCALING_COEFF);

  UnscentedKf::Transform sensorTf = unscentedSensorTransform(sigmaPts,
                                                             meanWeights,
                                                             covarianceWeights,
                                                             R);
  Eigen::VectorXd zPred = sensorTf.vector;  // Expected sensor vector
  Eigen::MatrixXd P_zz = sensorTf.covariance;  // Sensor/sensor covariance

  // Compute state/sensor cross-covariance
  UnscentedKf::SigmaPointSet predPointSet {x, sigmaPts};
  Eigen::MatrixXd predDeviations = computeDeviations(predPointSet);

  Eigen::MatrixXd P_xz = Eigen::MatrixXd::Zero(numStates, numStates);
  P_xz = predDeviations * covarianceWeights.asDiagonal()
      * sensorTf.deviations.transpose();

  // Compute Kalman gain
  Eigen::MatrixXd K = Eigen::MatrixXd::Zero(numStates, numStates);
  K = P_xz * P_zz.inverse();

  // Update state vector
  Eigen::VectorXd xCorr = Eigen::VectorXd::Zero(numStates);
  xCorr = x + K * (z - zPred);

  // Update state covariance
  Eigen::MatrixXd PCorr = Eigen::MatrixXd::Zero(numStates, numStates);
  PCorr = P - K * P_xz.transpose();
  //PCorr = PPred - K * P_zz * K.transpose()?

  UnscentedKf::Belief bel {xCorr, PCorr};
  return bel;
}

UnscentedKf::Transform UnscentedKf::unscentedStateTransform(
    Eigen::MatrixXd sigmaPts, Eigen::VectorXd meanWts, Eigen::VectorXd covWts,
    Eigen::MatrixXd noiseCov, double dt)
{
  int n = sigmaPts.rows();
  int L = sigmaPts.cols();
  Eigen::VectorXd vec = Eigen::VectorXd::Zero(n);
  Eigen::MatrixXd sigmas = Eigen::MatrixXd::Zero(n, L);
  Eigen::MatrixXd devs = Eigen::MatrixXd::Zero(n, L);
  Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(n, n);

  UnscentedKf::SigmaPointSet sample = sampleStateSpace(sigmaPts, meanWts, dt);
  vec = sample.vector;
  sigmas = sample.sigmaPoints;
  devs = computeDeviations(sample);
  cov = computeCovariance(devs, covWts, noiseCov);

  UnscentedKf::Transform out {vec, sigmas, cov, devs};
  return out;
}

UnscentedKf::Transform UnscentedKf::unscentedSensorTransform(
    Eigen::MatrixXd sigmaPts, Eigen::VectorXd meanWts, Eigen::VectorXd covWts,
    Eigen::MatrixXd noiseCov)
{
  int L = sigmaPts.cols();
  Eigen::VectorXd vec = Eigen::VectorXd::Zero(numStates);
  Eigen::MatrixXd sigmas = Eigen::MatrixXd::Zero(numStates, L);
  Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(numStates, numStates);
  Eigen::MatrixXd devs = Eigen::MatrixXd::Zero(numStates, L);

  UnscentedKf::SigmaPointSet sample = sampleSensorSpace(sigmaPts, meanWts);
  vec = sample.vector;
  sigmas = sample.sigmaPoints;
  devs = computeDeviations(sample);
  cov = computeCovariance(devs, covWts, noiseCov);

  UnscentedKf::Transform out {vec, sigmas, cov, devs};
  return out;
}

Eigen::MatrixXd UnscentedKf::computeSigmaPoints(const Eigen::VectorXd x,
                                                const Eigen::MatrixXd P,
                                                const double scalingCoeff) const
{
  // Compute lower Cholesky factor "A" of the given covariance matrix P
//  Eigen::LDLT<Eigen::MatrixXd> ldltOfCovMat(P); //TODO This causes rapid drift, but I'm not sure why. I have since switched from LDLT to LLT.
//  Eigen::MatrixXd L = ldltOfCovMat.matrixU().transpose();
  Eigen::LLT<Eigen::MatrixXd> lltOfCovMat(P);
  Eigen::MatrixXd L = lltOfCovMat.matrixL();
  Eigen::MatrixXd A = scalingCoeff * L;

  // Create a matrix "Y", which is then filled columnwise with the given
  // column vector x
  Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(numStates, numStates);
  Y = fillMatrixWithVector(x, numStates);

  // Create and populate sigma point matrix
  Eigen::MatrixXd sigmaPts(numStates, 2 * numStates + 1);
  sigmaPts << x, Y + A, Y - A;
  return sigmaPts;
}

Eigen::MatrixXd UnscentedKf::computeDeviations(
    UnscentedKf::SigmaPointSet sample) const
{
  Eigen::VectorXd vec = sample.vector;
  Eigen::MatrixXd sigmaPts = sample.sigmaPoints;

  int numRows = sigmaPts.rows();
  int numCols = sigmaPts.cols();
  Eigen::MatrixXd vecMat = fillMatrixWithVector(vec, numCols);
  Eigen::MatrixXd devs = Eigen::MatrixXd::Zero(numRows, numCols);

  return sigmaPts - vecMat;
}

Eigen::MatrixXd UnscentedKf::computeCovariance(Eigen::MatrixXd deviations,
                                               Eigen::VectorXd covWts,
                                               Eigen::MatrixXd noiseCov) const
{
  std::cout << "Deviations:\n" << deviations << std::endl;
  return deviations * covWts.asDiagonal() * deviations.transpose() + noiseCov;
}

UnscentedKf::SigmaPointSet UnscentedKf::sampleStateSpace(
    Eigen::MatrixXd sigmaPts, Eigen::VectorXd meanWts, double dt)
{
  int n = sigmaPts.rows();
  int L = sigmaPts.cols();

  Eigen::VectorXd vec = Eigen::VectorXd::Zero(n);
  Eigen::MatrixXd sigmas = Eigen::MatrixXd::Zero(n, L);
  for (int i = 0; i < L; i++)
  {
    sigmas.col(i) = processFunc(sigmaPts.col(i), dt);
    vec += meanWts(i) * sigmas.col(i);
  }

  UnscentedKf::SigmaPointSet out {vec, sigmas};
  return out;
}

UnscentedKf::SigmaPointSet UnscentedKf::sampleSensorSpace(
    Eigen::MatrixXd sigmaPts, Eigen::VectorXd meanWts)
{
  int L = sigmaPts.cols();

  Eigen::VectorXd vec = Eigen::VectorXd::Zero(numStates);
  Eigen::MatrixXd sigmas = Eigen::MatrixXd::Zero(numStates, L);
  for (int i = 0; i < L; i++)
  {
    sigmas.col(i) = observationFunc(sigmaPts.col(i).head(numSensors));
    vec = vec + meanWts(i) * sigmas.col(i);
  }

  UnscentedKf::SigmaPointSet out {vec, sigmas};
  return out;
}

Eigen::MatrixXd UnscentedKf::fillMatrixWithVector(Eigen::VectorXd vec,
                                                  int numCols) const
{
  int numRows = vec.rows();
  Eigen::MatrixXd mat(numRows, numCols);
  for (int i = 0; i < numCols; i++)
  {
    mat.col(i) = vec;
  }
  return mat;
}

void UnscentedKf::setWeights()
{
  // Set up mean weights
  meanWeights = Eigen::VectorXd::Zero(2 * numStates + 1);
  meanWeights(0) = LAMBDA / (numStates + LAMBDA);
  for (int i = 1; i < meanWeights.rows(); i++)
  {
    meanWeights(i) = 1 / (2 * (numStates + LAMBDA));
  }

  // Set up covariance weights
  covarianceWeights = meanWeights;
  covarianceWeights(0) += (1 - ALPHA * ALPHA + BETA);
}
