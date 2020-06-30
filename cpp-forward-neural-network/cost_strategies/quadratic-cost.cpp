#include "quadratic-cost.h"

Vector QuadraticCost::deltaOutput(const Vector& output, const Vector& expected,
		const Vector& non_sigmoid_output) {
	return std::move(coeffProduct(output - expected, 
		sigmoidPrime(non_sigmoid_output)));
}

double QuadraticCost::costFunction(const Vector& output, 
		const Vector& expected) {
	double cost = (output - expected).squaredNorm();
	return cost;
}