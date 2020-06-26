#include <eigen/Eigen/Dense>
#include <cmath>
#include <stdexcept>

using Eigen::Matrix;
using Vector = Eigen::VectorXd;

Vector sigmoid(Vector& z) {
	const int size = z.size();
	for (int i = 0; i != size; ++i)
		z(i) = 1 / (1 +  exp(z(i)));
	return z;
}

Vector sigmoidPrime(Vector& z) {
	return std::move(sigmoid(z) * (1 - sigmoid(z)));
}

Vector coeffProduct(const Vector& v1, const Vector& v2) {
	if (v1.size() != v2.size())
		throw std::invalid_argument("vectors not of same size");
	
	Matrix<double,  v1.size(), 1> product;
	for (int i = 0; i != v1.size(); ++i)
		product(i) = v1(i) * v2(i);

	return product;
}