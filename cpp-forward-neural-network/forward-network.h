#include <vector>
#include <array>
#include <eigen/Eigen/Dense>
#include <memory>

using Matrix = Eigen::MatrixXd;
using std::unique_ptr;
using std::shared_ptr;
using Vector = Eigen::VectorXd;

enum CostFunction {
	quadratic;
	crossentropy;
};


class ForwardNetwork {
public:
	ForwardNetwork(std::vector<int> layer_sizes, 
		CostFunction cost_type=quadratic);
	void SGD(std::vector<std::array<Vector, 2>>& training_data, 
		int epochs, int batch_size, double eta);
private:
	std::vector<Matrix> weights_;
	std::vector<Vector> biases_;
	const std::vector<int> layer_sizes_;
	unique_ptr<Cost> cost_;
	const int layers_;
};