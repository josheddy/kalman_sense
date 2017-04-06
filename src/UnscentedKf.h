#ifndef UNSCENTEDKF_H_
#define UNSCENTEDKF_H_

#include <Eigen/Dense>
#include <iostream> //TODO delete

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
  const double SIGMA_POINT_SCALING_COEFF = 3;//sqrt(numStates + LAMBDA);

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

  Transform unscentedStateTransform(const Eigen::MatrixXd sigmaPts,
                                    const Eigen::VectorXd meanWts,
                                    const Eigen::VectorXd covWts,
                                    const Eigen::MatrixXd measNoiseCov,
                                    const double dt);
  SigmaPointSet sampleStateSpace(const Eigen::MatrixXd sigmaPts,
                                 const Eigen::VectorXd meanWts,
                                 const double dt);

  Transform unscentedSensorTransform(const Eigen::MatrixXd sigmaPts,
                                     const Eigen::VectorXd meanWts,
                                     const Eigen::VectorXd covWts,
                                     const Eigen::MatrixXd noiseCov);
  SigmaPointSet sampleSensorSpace(const Eigen::MatrixXd sigmaPts,
                                  const Eigen::VectorXd meanWts);

  Eigen::MatrixXd computeCovariance(const Eigen::MatrixXd devs,
                                    const Eigen::VectorXd covWts,
                                    const Eigen::MatrixXd noiseCov) const;
  Eigen::MatrixXd computeDeviations(
      const UnscentedKf::SigmaPointSet sigmaPts) const;
  Eigen::MatrixXd computeSigmaPoints(const Eigen::VectorXd x,
                                     const Eigen::MatrixXd P,
                                     const double scalingCoeff) const;
  Eigen::MatrixXd fillMatrixWithVector(const Eigen::VectorXd vec,
                                       const int numCols) const;
};

#endif  // UNSCENTEDKF_H_
