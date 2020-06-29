#pragma once
#include <eigen/Eigen/Dense>

using Vector = Eigen::VectorXd;

/* Interface for the cost function class. */
class Cost {
public:
	virtual Vector delta_output(const Vector& non_sigmoid_output,
		const Vector& output, const Vector& expected) = 0;
	virtual double cost_function(const Vector& output, 
		const Vector& expected) = 0;
};