#include "forward-network.h"
#include <algorithm>	// for std::random_shuffle
#include <stdexcept>
#include <iostream>
#include <random>

using Matrix = Eigen::MatrixXd;
using std::unique_ptr;
using std::shared_ptr;
using Vector = Eigen::VectorXd;

ForwardNetwork::ForwardNetwork(std::vector<int> layer_sizes, 
		CostFunction cost_type, DataType data_type) : 
		layers_{layer_sizes.size()}, layer_sizes_{layer_sizes} {
	for (int i = 0; i != layers_- 1; ++i) {
		// set gaussian distribution with mean 0 and standard deviation 1
		std::random_device rd;
		std::default_random_engine generator(rd());
		std::normal_distribution<double> distribution{0, 1};
		auto normal = [&] (double) {return distribution(generator);};
		// initialize weights and biases randomly
		biases_.push_back(
			Vector::NullaryExpr(layer_sizes[i+1], normal)
		);
		weights_.push_back(
			Matrix::NullaryExpr(layer_sizes[i+1], layer_sizes[i], normal)
		);
		// std::cout << "weights: " << weights_[0] << std::endl;
		// std::cout << "biases: " << biases_[0] << std::endl;
	}
	switch (cost_type) {
		case crossentropy:
			cost_= std::make_unique<CrossEntropyCost>();
			break;
		default:
			cost_= std::make_unique<QuadraticCost>();
	}
	switch (data_type) {
		case binary:
			data_ = std::make_unique<ReadMNist>();
			break;
		default:
			data_ = std::make_unique<ReadText>();
	}
}

void ForwardNetwork::dataSource(const std::vector<std::string>& files, 
		bool training) const {
	if (files.size() == 1) 
		data_->read(files[0], training);	// read training data
	else if (files.size() == 2) {
		data_->readData(files[0], training);
		data_->readLabel(files[1], training);
	}
	else
		throw std::invalid_argument(
			"program not compatible with current number of files"
		);

}

int ForwardNetwork::trainingSize() const {
	if (data_->training_data_.empty())
		throw std::out_of_range("training data has not been set");
	return data_->training_data_.size();
}

int ForwardNetwork::testSize() const {
	if (data_->test_data_.empty())
		throw std::out_of_range("test data has not been set");
	return data_->test_data_.size();
}

void ForwardNetwork::SGD(int epochs, int batch_size, double eta, bool test) {
	for (int epoch = 0; epoch != epochs; ++epoch) {
		std::cout << "current epoch: " << epoch << std::endl;
		// shuffle the data
		std::random_shuffle(data_->training_data_.begin(), 
			data_->training_data_.end());
		// divide the training data in batches of size batch_size
		int nr_of_batches = data_->training_data_.size() / batch_size;
		// keep track of training example
		int train_ex = 0;	
		for (int batch = 0; batch != nr_of_batches; ++batch) {
			// for every training example in the batch, we have a vector
			// of Vectors, where every Vector contains a layer of 
			// activations/weighted inputs/deltas, respectively
			std::vector<std::vector<Vector>> activations(batch_size);
			std::vector<std::vector<Vector>> w_inputs(batch_size);
			std::vector<std::vector<Vector>> delta(batch_size);
			// for each training example, apply backpropagation 
			for (int exb = 0; exb != batch_size; ++exb) {
				// feedforward to calculate activations and weighted inputs
				feedForward(activations[exb], w_inputs[exb], train_ex);
				// determine deltas with backpropagation
				backProp(activations[exb], w_inputs[exb],  
					delta[exb], data_->training_data_[train_ex][1]);
				++train_ex;
			}
			// update weights and biases using (ch 1, 20), (ch 1, 21),
			// (ch 2, BP3), (ch2, BP4)
			double step = eta / batch_size;
			for (int lyr = 0; lyr != layers_ - 1; ++lyr) {
				Matrix weight_summand = 
					Matrix::Zero(weights_[lyr].rows(), weights_[lyr].cols());
				Vector bias_summand = 
					Vector::Zero(biases_[lyr].size());
				for (int exb = 0; exb != batch_size; ++exb) {
					weight_summand += 
						delta[exb][lyr] * activations[exb][lyr].transpose();
					bias_summand += delta[exb][lyr];
				}
				weights_[lyr] = weights_[lyr] - step*weight_summand;
				biases_[lyr] = biases_[lyr] - step*bias_summand; 
			}
			if (test)
				this->test(false);
		}
	}
}

double ForwardNetwork::test(bool test_data) const {
	double total_cost = 0;
	int size = trainingSize();
	if (test_data)
		size = testSize();
	std::vector<std::vector<Vector>> activations(size);
	std::vector<std::vector<Vector>> w_inputs(size);
	for (int ex = 0; ex != size; ++ex) {
		activations[ex].reserve(layers_);
		w_inputs[ex].reserve(layers_ - 1);
		feedForward(activations[ex], w_inputs[ex], ex);
		// add the cost for this example to the total cost
		if (test_data) 
			total_cost += cost_->costFunction(
				activations[ex][layers_-1], data_->test_data_[ex][1]
			);
		else 	// test on training data if test data is not available
			total_cost += cost_->costFunction(
				activations[ex][layers_-1], data_->training_data_[ex][1]
			);
	}
	std::cout << "cost for this batch: " << total_cost / size 
		<< std::endl;
	return total_cost / size;
}

void ForwardNetwork::resetNetwork() {
	biases_ = std::vector<Vector>();
	weights_ = std::vector<Matrix>();
	biases_.reserve(layers_-1);
	weights_.reserve(layers_-1);
	for (int i = 0; i != layers_- 1; ++i) {
		// set gaussian distribution with mean 0 and standard dev 1
		std::random_device rd;
		std::default_random_engine generator(rd());
		std::normal_distribution<double> distribution{0, 1};
		auto normal = [&] (double) {return distribution(generator);};
		//auto normal2 = [&] (double) {return distribution(generator);};
		// initialize weights and biases randomly
		biases_.push_back(
			Vector::NullaryExpr(layer_sizes_[i+1], normal)
		);
		weights_.push_back(
			Matrix::NullaryExpr(layer_sizes_[i+1], layer_sizes_[i], normal)
		);
	}
	// std::cout << "weights: " << weights_[0] << std::endl;
	// std::cout << "biases: " << biases_[0] << std::endl;
}

void ForwardNetwork::feedForward(std::vector<Vector>& activations,
		std::vector<Vector>& w_inputs, int train_ex) const {
	// reserve enough memory to prevent many resize operations
	activations.reserve(layers_);
	w_inputs.reserve(layers_-1);
	// activate first layer
	activations.push_back(data_->training_data_[train_ex][0]);
	// calculated weighted inputs and activations
	for (int lyr = 0; lyr != layers_ - 1; ++lyr) {
		w_inputs.push_back( 
			weights_[lyr]*(activations[lyr]) + biases_[lyr]
		);
		activations.push_back(sigmoid(w_inputs[lyr]));
	}
}

void ForwardNetwork::backProp(const std::vector<Vector>& activations, 
		const std::vector<Vector>& w_inputs,
		std::vector<Vector>& delta, const Vector& output) const {
	delta.resize(layers_ - 1);
	delta[layers_ - 2] = 
		cost_->deltaOutput(activations[layers_-1], output,
			w_inputs[layers_-2]
		);
	for (int lyr = layers_ - 3; lyr > -1; --lyr)
		delta[lyr] = coeffProduct(
			weights_[lyr+1].transpose() * delta[lyr+1], 
			sigmoidPrime(w_inputs[lyr])	// (ch2, BP2)
		);
}



