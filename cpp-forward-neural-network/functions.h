#pragma once
#include <eigen/Eigen/Dense>
#include <cmath>	// for exp

using Vector = Eigen::VectorXd;

/* sigmoid function as per (ch 1, 3) */
inline Vector sigmoid(const Vector& z) {
	return z.unaryExpr([](double x){return 1.0 / (1.0 + exp(-x));});
}

/* the derivative of the sigmoid function */
inline Vector sigmoidPrime(const Vector& z) {
	return sigmoid(z) - sigmoid(z).cwiseProduct(sigmoid(z));
}
