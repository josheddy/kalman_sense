// UnscentedKf.h

#ifndef UNSCENTEDKF_H_
#define UNSCENTEDKF_H_

#include <Eigen/Dense>

class UnscentedKf
{
public:
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
  struct Belief
  {
    Eigen::VectorXd state;
    Eigen::MatrixXd covariance;
  };

  // Asynchronous prediction and correction methods
  UnscentedKf::Transform predictState(Eigen::VectorXd x, Eigen::MatrixXd P,
                                      Eigen::MatrixXd Q, double dt);
  UnscentedKf::Belief correctState(UnscentedKf::Transform stateTf,
                                   Eigen::VectorXd z, Eigen::MatrixXd R);

  // Synchronous, all-in-one prediction/correction method
  UnscentedKf::Belief run(Eigen::VectorXd x, Eigen::MatrixXd P,
                          Eigen::VectorXd z, Eigen::MatrixXd Q,
                          Eigen::MatrixXd R, double dt);

  virtual Eigen::VectorXd processFunc(Eigen::VectorXd x, double dt) = 0;
  virtual Eigen::VectorXd observationFunc(Eigen::VectorXd z) = 0;
  virtual ~UnscentedKf() = 0;
  UnscentedKf();

  const int numStates;

  // Tunable parameters
  const double alpha = 0.0003;
  const double kappa = 0;
  const double beta = 2;
  const double lambda = (alpha * alpha) * (numStates + kappa) - numStates;
  Eigen::VectorXd meanWeights, covarianceWeights;

  Transform unscentedStateTransform(Eigen::MatrixXd sigmaPts,
                                    Eigen::VectorXd meanWts,
                                    Eigen::VectorXd covWts,
                                    Eigen::MatrixXd measNoiseCov, double dt);
  SigmaPointSet sampleStateSpace(Eigen::MatrixXd sigmaPts,
                                 Eigen::VectorXd meanWts, double dt);

  Transform unscentedSensorTransform(int numSensors, Eigen::MatrixXd sigmaPts,
                                     Eigen::VectorXd meanWts,
                                     Eigen::VectorXd covWts,
                                     Eigen::MatrixXd noiseCov);
  SigmaPointSet sampleSensorSpace(int numSensors, Eigen::MatrixXd sigmaPts,
                                  Eigen::VectorXd meanWts);

  Eigen::MatrixXd computeCovariance(Eigen::MatrixXd devs,
                                    Eigen::VectorXd covWts,
                                    Eigen::MatrixXd noiseCov) const;
  Eigen::MatrixXd computeDeviations(UnscentedKf::SigmaPointSet sigmaPts) const;
  Eigen::MatrixXd computeSigmaPoints(Eigen::VectorXd x, Eigen::MatrixXd P,
                                     double scalingCoeff) const;
  Eigen::MatrixXd fillMatrixWithVector(Eigen::VectorXd vec, int numCols) const;
};

#endif /* UNSCENTEDKF_H_ */
