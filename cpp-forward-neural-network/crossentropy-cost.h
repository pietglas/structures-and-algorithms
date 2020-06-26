#include "cost.h"
#include <eigen/Eigen/Dense>
using Eigen::Dynamic;
using Vector = Matrix<double, Dynamic, 1>;

class CrossEntropyCost : public Cost {
public:
	virtual Vector<size> delta_output(const Vector<size>& output,
		const Vector<size>& expected) override;
	virtual double cost_function(const Vector<size>& output,
		const Vector<size>& expected) override;
private:
	
};