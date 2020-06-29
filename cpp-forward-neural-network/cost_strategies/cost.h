#pragma once
#include <eigen/Eigen/Dense>

using Vector = Eigen::VectorXd;

/* Interface for the cost function class. */
class Cost {
public:
	/* returns delta^L = nabla_C / nabla_z^L, where L is the 
	   last layer in the network */ 
	virtual Vector deltaOutput(const Vector& non_sigmoid_output,
		const Vector& output, const Vector& expected) = 0;
	/* returns the cost for a given output */
	virtual double costFunction(const Vector& output, 
		const Vector& expected) = 0;
};