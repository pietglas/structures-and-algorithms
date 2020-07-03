#pragma once
#include <eigen/Eigen/Dense>
#include <cmath>

using Vector = Eigen::VectorXd;

/* sigmoid function as per (ch 1, 3) */
inline Vector sigmoid(const Vector& z) {
	return z.unaryExpr([](double x){return 1 / (1 + exp(-x));});
}

/* the derivative of the sigmoid function */
inline Vector sigmoidPrime(const Vector& z) {
	return sigmoid(z) - sigmoid(z).cwiseProduct(sigmoid(z));
}
