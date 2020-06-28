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

/* Class that contains the data and methods for a neural network. The SGD
 * methods implements stochastic gradient descent to train the network. 
 * It is initialized with a vector indicating the size of every layer, e.g.
 * {2, 3, 1} gives a network with two input neurons, one middle layer with
 * 3 neurons and 1 output neuron. Optionally, one can specify the cost
 * function that is to be used (see the enum above). 
 */
class ForwardNetwork {
public:
	ForwardNetwork(std::vector<int> layer_sizes, 
		CostFunction cost_type=quadratic);
	/* Stochastic Gradient Descent algorithm, where `training_data` is a
	 * vector of arrays containing the input and expected output, 
	 * `epochs` the number of times we loop over the training data, 
	 * `batch_size` the number of training examples to which we apply
	 * backpropagation, and `eta` the learning rate. 
	 */
	void SGD(std::vector<std::array<Vector, 2>>& training_data, 
		int epochs, int batch_size, double eta);
private:
	std::vector<Matrix> weights_;
	std::vector<Vector> biases_;
	const std::vector<int> layer_sizes_;
	unique_ptr<Cost> cost_;
	const int layers_;

	/* determines the deltas for each layer, using backpropagation */
	std::vector<Vector> backProp(const std::vector<Vector>& w_inputs,
		const std::vector<Vector>& activations, int ex) const
};