#include "quadratic-cost.h"

Vector QuadraticCost::delta_output(const Vector& non_sigmoid_output,
		const Vector& output, const Vector& expected) {
	return std::move(coeffProduct(output - expected, 
		sigmoidPrime(non_sigmoid_output)));
}

double QuadraticCost::cost_function(const Vector& output, 
		const Vector& expected) {
	double cost = (output - expected).squaredNorm();
	return cost;
}