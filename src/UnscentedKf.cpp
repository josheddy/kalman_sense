#include "UnscentedKf.h"
#include <iostream>

UnscentedKf::UnscentedKf() :
    numStates(16)
{
}

UnscentedKf::~UnscentedKf()
{
}

UnscentedKf::Belief UnscentedKf::predictState(Eigen::VectorXd x,
                                              Eigen::MatrixXd P,
                                              Eigen::MatrixXd Q, double dt)
{
  // Compute sigma points around current estimated state
  int n = x.rows();
  double scalingCoeff = pow(n + lambda, 0.5);
  Eigen::MatrixXd sigmaPts(n, 2 * n + 1);
  sigmaPts = computeSigmaPoints(x, P, scalingCoeff);

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
  int n = x.rows();
  double scalingCoeff = n + lambda;
  Eigen::MatrixXd sigmaPts(n, 2 * n + 1);
  sigmaPts = computeSigmaPoints(x, P, scalingCoeff);

  int m = z.rows();
  UnscentedKf::Transform sensorTf = unscentedSensorTransform(m, sigmaPts,
                                                             meanWeights,
                                                             covarianceWeights,
                                                             R);
  Eigen::VectorXd zPred = sensorTf.vector;  // Expected sensor vector
  Eigen::MatrixXd P_zz = sensorTf.covariance;  // Sensor/sensor covariance

  // Compute state/sensor cross-covariance
  UnscentedKf::SigmaPointSet predPointSet {x, sigmaPts};
  Eigen::MatrixXd predDeviations = computeDeviations(predPointSet);
  Eigen::MatrixXd P_xz = Eigen::MatrixXd::Zero(n, m);
  P_xz = predDeviations * covarianceWeights.asDiagonal()
      * sensorTf.deviations.transpose();

  // Compute Kalman gain
  Eigen::MatrixXd K = Eigen::MatrixXd::Zero(n, m);
  K = P_xz * P_zz.inverse();

  // Update state vector
  Eigen::VectorXd xCorr = Eigen::VectorXd::Zero(n);
  xCorr = x + K * (z - zPred);

  // Update state covariance
  Eigen::MatrixXd PCorr = Eigen::MatrixXd::Zero(n, n);
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

  UnscentedKf::SigmaPointSet sample = sampleStateSpace(sigmaPts, meanWts, dt);

  vec = sample.vector;
  sigmas = sample.sigmaPoints;
  std::cout << "vec from state UT\n" << vec << std::endl;
  //std::cout << "sigma pts from state UT\n" << sigmas << std::endl;

  Eigen::MatrixXd devs = Eigen::MatrixXd::Zero(n, L);
  devs = computeDeviations(sample);
  //std::cout << "deviations from state UT\n" << devs << std::endl;

  Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(n, n);
  cov = computeCovariance(devs, covWts, noiseCov);
  //std::cout << "cov mat from state UT\n" << cov << std::endl;

  UnscentedKf::Transform out {vec, sigmas, cov, devs};
  return out;
}

UnscentedKf::Transform UnscentedKf::unscentedSensorTransform(
    int numSensors, Eigen::MatrixXd sigmaPts, Eigen::VectorXd meanWts,
    Eigen::VectorXd covWts, Eigen::MatrixXd noiseCov)
{
  int n = sigmaPts.rows();
  int L = sigmaPts.cols();
  Eigen::VectorXd vec = Eigen::VectorXd::Zero(n);
  Eigen::MatrixXd sigmas = Eigen::MatrixXd::Zero(n, L);
  Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(n, n);
  Eigen::MatrixXd devs = Eigen::MatrixXd::Zero(n, L);

  UnscentedKf::SigmaPointSet sample = sampleSensorSpace(numSensors, sigmaPts,
                                                        meanWts);

  vec = sample.vector;
  sigmas = sample.sigmaPoints;

  devs = computeDeviations(sample);
  cov = computeCovariance(devs, covWts, noiseCov);

  UnscentedKf::Transform out {vec, sigmas, cov, devs};
  return out;
}

Eigen::MatrixXd UnscentedKf::computeSigmaPoints(Eigen::VectorXd x,
                                                Eigen::MatrixXd P,
                                                double scalingCoeff)
{
  // Compute lower Cholesky factor "A" of the given covariance matrix P
  //Eigen::LDLT<Eigen::MatrixXd> ldltOfCovMat(P);
  //Eigen::MatrixXd L = ldltOfCovMat.matrixL();
  Eigen::LLT<Eigen::MatrixXd> lltOfCovMat(P);
  Eigen::MatrixXd L = lltOfCovMat.matrixL().transpose();
  Eigen::MatrixXd A = scalingCoeff * L;

  // Create a matrix "Y", which is then filled columnwise with the given
  // column vector x
  int n = x.rows();
  Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(n, n);
  Y = fillMatrixWithVector(x, n);

  // Create and populate sigma point matrix
  Eigen::MatrixXd sigmaPts(n, 2 * n + 1);
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
    int numSensors, Eigen::MatrixXd sigmaPts, Eigen::VectorXd meanWts)
{
  int n = numSensors;
  int L = sigmaPts.cols();

  Eigen::VectorXd vec = Eigen::VectorXd::Zero(n);
  Eigen::MatrixXd sigmas = Eigen::MatrixXd::Zero(n, L);
  for (int i = 0; i < L; i++)
  {
    sigmas.col(i) = observationFunc(sigmaPts.col(i));
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
