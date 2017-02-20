#ifndef UNSCENTEDKF_H_
#define UNSCENTEDKF_H_

#include <Eigen/Dense>

class UnscentedKf
{
public:
  UnscentedKf();
  virtual ~UnscentedKf() = 0;

  struct Belief
  {
    Eigen::VectorXd state;
    Eigen::MatrixXd covariance;
  };

  UnscentedKf::Belief predictState(Eigen::VectorXd x, Eigen::MatrixXd P,
                                   Eigen::MatrixXd Q, double dt);
  UnscentedKf::Belief correctState(Eigen::VectorXd x, Eigen::MatrixXd P,
                                   Eigen::VectorXd z, Eigen::MatrixXd R);

  void setWeights();
  int numStates;
  int numSensors;

private:
  Eigen::VectorXd meanWeights, covarianceWeights;

  // Tunable parameters
  const double ALPHA = 0.001;
  const double BETA = 2;
  const double KAPPA = 0;

  const double LAMBDA = pow(ALPHA, 2) * (numStates + KAPPA) - numStates;
  const double SIGMA_PT_SCALING_COEFF = pow(numStates + LAMBDA, 0.5);

  virtual Eigen::VectorXd processFunc(Eigen::VectorXd x, double dt) = 0;
  virtual Eigen::VectorXd observationFunc(Eigen::VectorXd z) = 0;

  struct Transform
  {
    Eigen::VectorXd vector;
    Eigen::MatrixXd sigmaPoints;
    Eigen::MatrixXd covariance;
    Eigen::MatrixXd deviations;
  };

  struct SigmaPointSet
  {
    Eigen::VectorXd vector;
    Eigen::MatrixXd sigmaPoints;
  };

  Transform unscentedStateTransform(Eigen::MatrixXd sigmaPts,
                                    Eigen::VectorXd meanWts,
                                    Eigen::VectorXd covWts,
                                    Eigen::MatrixXd measNoiseCov, double dt);
  SigmaPointSet sampleStateSpace(Eigen::MatrixXd sigmaPts,
                                 Eigen::VectorXd meanWts, double dt);

  Transform unscentedSensorTransform(Eigen::MatrixXd sigmaPts,
                                     Eigen::VectorXd meanWts,
                                     Eigen::VectorXd covWts,
                                     Eigen::MatrixXd noiseCov);
  SigmaPointSet sampleSensorSpace(Eigen::MatrixXd sigmaPts,
                                  Eigen::VectorXd meanWts);

  Eigen::MatrixXd computeCovariance(Eigen::MatrixXd devs,
                                    Eigen::VectorXd covWts,
                                    Eigen::MatrixXd noiseCov) const;
  Eigen::MatrixXd computeDeviations(UnscentedKf::SigmaPointSet sigmaPts) const;
  Eigen::MatrixXd computeSigmaPoints(const Eigen::VectorXd x,
                                     const Eigen::MatrixXd P,
                                     const double scalingCoeff) const;
  Eigen::MatrixXd fillMatrixWithVector(Eigen::VectorXd vec, int numCols) const;
};

#endif /* UNSCENTEDKF_H_ */
