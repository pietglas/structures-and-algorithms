#include "cost.h"
#include <eigen/Eigen/Dense>
using Eigen::Dynamic;
using Vector = Matrix<double, Dynamic, 1>;

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
	virtual double cost_function(const Vector<size>& output,
		const Vector<size>& expected) override;
private:
	
};

Vector CrossEntropyCost::delta_output(const Vector& non_sigmoid_output,
		const Vector& output, const Vector& expected) {
	return std::move(output - expected);
}

// Currently the cost is not well defined (for output values equal
// to either 0 or 1), and it is probably not well-behaved for output
// values close to 0. 
double CrossEntropyCost::cost_function(const Vector& output, 
		const Vector& expected) {
	double cost = 0;
	for (int i = 0; i != output.size(); ++i)
		sum += expected(i)*log(output(i)) + 
			(1 - expected(i))*log(1 - output(i));
	return sum;
}
