// SimpleUkf.h

#ifndef SIMPLEUKF_H_
#define SIMPLEUKF_H_

#include <Eigen/Dense>
#include "UnscentedKf.h"

class SimpleUkf : public UnscentedKf
{
public:
  int numStates;
  int numSensors;

  SimpleUkf(Eigen::VectorXd initStateVec, Eigen::VectorXd initSensorVec);
  virtual ~SimpleUkf();
  Eigen::VectorXd processFunc(Eigen::VectorXd stateVec);
  Eigen::VectorXd observationFunc(Eigen::VectorXd stateVec);
};

#endif /* SIMPLEUKF_H_ */
