#include "cost.h"
#include <eigen/Eigen/Dense>

template<int size>
class QuadraticCost : public Cost {
public:
	virtual Vector<size> delta_output(const Vector<size>& output,
		const Vector<size>& expected) override;
	virtual double cost_function(const Vector<size>& output,
		const Vector<size>& expected) override;
private:
	
};


