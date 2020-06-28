#include <eigen/Eigen/Dense>
#include <cmath>
#include <stdexcept>

using Eigen::Matrix;
using Vector = Eigen::VectorXd;

/* sigmoid function as per (ch 1, 3) */
Vector sigmoid(const Vector& z) {
	Vector result(z.size());
	for (int i = 0; i != z.size(); ++i)
		result(i) = 1 / (1 +  exp(z(i)));
	return result;
}

/* the derivative of the sigmoid function */
Vector sigmoidPrime(Vector& z) {
	return std::move(sigmoid(z) * (1 - sigmoid(z)));
}

/* calculates the coefficient-wise product of two column vectors
   of equal size. Throws std::invalid_argument if the vectors
   do not have the same size */
Vector coeffProduct(const Vector& v1, const Vector& v2) {
	if (v1.size() != v2.size())
		throw std::invalid_argument("vectors not of same size");
	Matrix<double,  v1.size(), 1> product;
	for (int i = 0; i != v1.size(); ++i)
		product(i) = v1(i) * v2(i);

	return product;
}