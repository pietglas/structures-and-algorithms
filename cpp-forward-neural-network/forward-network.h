#pragma once
#include <vector>
#include <array>
#include <string>
#include <eigen/Eigen/Dense>
#include <memory>
#include "cost_strategies/cost.h"
#include "cost_strategies/quadratic-cost.h"
#include "cost_strategies/crossentropy-cost.h"
#include "data_strategies/read-data.h"
#include "data_strategies/read-text.h"
#include "data_strategies/read-mnist.h"

using Matrix = Eigen::MatrixXd;
using std::unique_ptr;
using std::shared_ptr;
using Vector = Eigen::VectorXd;

enum CostFunction {
	quadratic,
	crossentropy
};

enum DataType {
	binary,
	text
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
		CostFunction cost_type=quadratic, DataType data_type=text);
	void dataSource(const std::vector<std::string>& files, bool training) const;
	/* Stochastic Gradient Descent algorithm, where `training_data` is a
	 * vector of arrays containing the input and expected output, 
	 * `epochs` the number of times we loop over the training data, 
	 * `batch_size` the number of training examples to which we apply
	 * backpropagation, where -1 indicates regular gradient descent,
	 * and `eta` the learning rate. 
	 */
	void SGD(const std::vector<std::string>& data, 
		int epochs, int batch_size=-1, double eta=0.5);
private:
	std::vector<Matrix> weights_;
	std::vector<Vector> biases_;
	const std::vector<int> layer_sizes_;
	unique_ptr<Cost> cost_;
	unique_ptr<ReadData> data_;
	const int layers_;

	/* determines the deltas for each layer, using backpropagation */
	std::vector<Vector> backProp(const std::vector<Vector>& w_inputs,
		const std::vector<Vector>& activations, const Vector& output) const;
};