#include <eigen/Eigen/Dense>
using Eigen::Dynamic = Dynamic;
using Vector = Matrix<double, Dynamic, 1>;

class Cost {
	virtual Vector delta_output(const Vector& non_sigmoid_output,
		const Vector& output, const Vector& expected) = 0;
	virtual double cost_function(const Vector& output, 
		const Vector& expected) = 0;
}