#pragma once
#include <eigen/Eigen/Dense>
#include <cmath>
#include <stdexcept>
//#include <omp.h>

using Vector = Eigen::VectorXd;

/* calculates the coefficient-wise product of two column vectors
   of equal size. Throws std::invalid_argument if the vectors
   do not have the same size */
inline Vector coeffProduct(const Vector& v1, const Vector& v2) {
	if (v1.size() != v2.size())
		throw std::invalid_argument("vectors not of same size");
	const int size = v1.size();
	Vector product(size);
	for (int i = 0; i < size; ++i)
		product(i) = v1(i) * v2(i);

	return product;
}

/* sigmoid function as per (ch 1, 3) */
inline Vector sigmoid(const Vector& z) {
	Vector result(z.size());
	const int size = z.size();
	// omp_set_num_threads(4);
	// #pragma omp parallel for
	for (int i = 0; i < size; ++i)
		result(i) = 1 / (1 +  exp(-z(i)));
	return result;
}

/* the derivative of the sigmoid function */
inline Vector sigmoidPrime(const Vector& z) {
	return sigmoid(z) - coeffProduct(sigmoid(z), sigmoid(z));
}
