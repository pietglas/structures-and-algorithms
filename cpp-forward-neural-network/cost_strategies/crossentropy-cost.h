#pragma once
#include "cost.h"
#include <eigen/Eigen/Dense>

using Vector = Eigen::VectorXd;

/** 
 * Class that wraps the cross-entropy loss function, as well as
 * the value of `delta` in the final layer, for this function 
 * (See chapter 3, equation (68) of Nielsens book). 
 */
class CrossEntropyCost : public Cost {
public:
	virtual Vector delta_output(const Vector& non_sigmoid_output,
		const Vector& output, const Vector& expected) override;
	/* cross-entropy cost function as per (ch 3, 57) */
	virtual double cost_function(const Vector& output,
		const Vector& expected) override;
private:
	
};
