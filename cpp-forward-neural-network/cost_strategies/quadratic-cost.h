#pragma once 
#include "cost.h"
#include "../functions.h"
#include <eigen/Eigen/Dense>

using Vector = Eigen::VectorXd;

/** 
 * Class that wraps the quadratic cost loss function, as well as
 * the value of `delta` in the final layer, for this function 
 * (See chapter 1, equation (30) of Nielsens book). 
 */
class QuadraticCost : public Cost {
public:
	virtual Vector delta_output(const Vector& non_sigmoid_output,
		const Vector& output, const Vector& expected) override;
	/* quadratic cost function as per (ch 1, 6) */
	virtual double cost_function(const Vector& output,
		const Vector& expected) override;
private:
	
};


