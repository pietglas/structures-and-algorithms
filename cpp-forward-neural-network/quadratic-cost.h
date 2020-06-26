#include "cost.h"
#include "functions.h"
#include <eigen/Eigen/Dense>
using Eigen::Dynamic;
using Vector = Matrix<double, Dynamic, 1>;

class QuadraticCost : public Cost {
public:
	virtual Vector delta_output(const Vector& non_sigmoid_output,
		const Vector& output, const Vector& expected) override;
	virtual double cost_function(const Vector& output,
		const Vector& expected) override;
private:
	
};

Vector QuadraticCost::delta_output(const Vector& non_sigmoid_output,
		const Vector& output, const Vector& expected) {
	return std::move(coeffProduct(output - expected, 
		sigmoid_prime(non_sigmoid_output)));
}

double QuadraticCost::cost_function(const Vector& output, 
		const Vector& expected) {
	return std::move(squaredNorm(output - expected));
}


