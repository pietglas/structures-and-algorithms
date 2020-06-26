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
	ForwardNetwork(std::vector<int> layer_sizes, CostFunction cost_type);
	void SGD(std::vector<shared_ptr<Vector>> training_data, int epochs,
		int batch_size, double eta);
private:
	std::vector<shared_ptr<Matrix>> weights_;
	std::vector<shared_ptr<Vector>> biases_;
	unique_ptr<Cost> cost_function_;
	int layers_;
	// returns a vector with the delta vector of each layer
	std::vector<Vector> backProp(const Vector& input,
		const Vector& output) const;
	// returns a vector with the activation layers
	std::vector<Vector> activateLayers(const Vector& input) const;
};