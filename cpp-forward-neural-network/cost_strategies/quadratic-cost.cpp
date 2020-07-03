#include "quadratic-cost.h"

Vector QuadraticCost::deltaOutput(const Vector& output, const Vector& expected,
		const Vector& non_sigmoid_output) {
	return (output - expected).cwiseProduct(sigmoidPrime(non_sigmoid_output));
}

double QuadraticCost::costFunction(const Vector& output, 
		const Vector& expected) {
	return (output - expected).squaredNorm();
}