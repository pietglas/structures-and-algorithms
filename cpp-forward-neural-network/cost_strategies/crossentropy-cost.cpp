#include "crossentropy-cost.h"

inline Vector CrossEntropyCost::deltaOutput(const Vector& non_sigmoid_output,
		const Vector& output, const Vector& expected) {
	return output - expected;
}

// Currently the cost is not well defined (for output values equal
// to either 0 or 1), and it is probably not well-behaved for output
// values close to 0. 
double CrossEntropyCost::costFunction(const Vector& output, 
		const Vector& expected) {
	double cost = 0;
	for (int i = 0; i != output.size(); ++i)
		cost += expected(i)*log(output(i)) + 
			(1 - expected(i))*log(1 - output(i));
	return cost;
}