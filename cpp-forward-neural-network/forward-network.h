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
	text,
	mnist
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
	/* set the data. If `training`, the source is assumed to be a
	 * training set. If no file is provided, the data is set to 
	 * the standard choice
	 */ 
	void data(bool training, const std::string& file=std::string(""));
	/* returns the size of the dataset with training examples. Throws
	   std::out_of_range if training data is not set */
	int trainingSize() const;
	/* returns the size of the dataset with test examples. Throws 
	   std::out_of_range if the test data has not been set */
	int testSize() const;
	/* allows the user to switch between several cost functions
	   and data formats. */
	void setDataType(const DataType& data_type);
	void setCost(const CostFunction& cost_function);
	/* Stochastic Gradient Descent algorithm, where `training_data` is a
	 * vector of arrays containing the input and expected output, 
	 * `epochs` the number of times we loop over the training data, 
	 * `batch_size` the number of training examples to which we apply
	 * backpropagation, where -1 indicates regular gradient descent,
	 * and `eta` the learning rate. 
	 */
	void SGD(int epochs, int batch_size, double eta=0.5, bool test=false);
	/* test the network performance. If one doesn't have seperate test data
	   provide `false` as an argument */
	void test(bool test_data=true);
	/* resets the weights and biases in the network to randomized state */ 
	void resetNetwork();
private:
	const int layers_;
	const std::vector<int> layer_sizes_;
	std::vector<Matrix> weights_;
	std::vector<Vector> biases_;
	unique_ptr<Cost> cost_;
	unique_ptr<ReadData> data_;
	std::string training_data_;
	std::string test_data_;

	void feedForward(std::vector<Vector>& activations,
		std::vector<Vector>& w_inputs, int train_ex) const;
	/* determines the deltas for each layer, using backpropagation */
	void backProp(const std::vector<Vector>& activations, 
		const std::vector<Vector>& w_inputs, std::vector<Vector>& delta, 
		const Vector& expected) const;
	void setWeightsBiasesRandom();
};