#include "forward-network.h"
#include <algorithm>	// for std::random_shuffle
#include <stdexcept>

using Matrix = Eigen::MatrixXd;
using std::unique_ptr;
using std::shared_ptr;
using Vector = Eigen::VectorXd;

ForwardNetwork::ForwardNetwork(std::vector<int> layer_sizes, 
		CostFunction cost_type, DataType data_type) : layers_{layer_sizes.size()},
		layer_sizes_{layer_sizes} {
	for (int i = 0; i != layers_- 2; ++i) {
		biases_[i] = Vector::Random(layer_sizes[i]);
		weights_[i-1] = Matrix::Random(layer_sizes[i], layer_sizes[i-1]);
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

void ForwardNetwork::SGD(int epochs, int batch_size, double eta) {
	for (int epoch = 0; epoch != epochs; ++epoch) {
		// shuffle the data
		std::random_shuffle(data_->training_data_.begin(), data_->training_data_.end());
		// divide the training data in batches of size batch_size
		int nr_of_batches = data_->training_data_.size() / batch_size;
		for (int batch = 0; batch != nr_of_batches; ++batch) {
			// for every training example in the batch, we have a vector
			// of Vectors, where every Vector contains a layer of 
			// activations/weighted inputs/deltas, respectively
			std::vector<std::vector<Vector>> activations(batch_size);
			std::vector<std::vector<Vector>> w_inputs(batch_size);
			std::vector<std::vector<Vector>> delta;
			delta.reserve(batch_size);
			// for each training example, apply backpropagation 
			for (int exb = 0; exb != batch_size; ++exb) {
				// set training example input
				activations[exb].push_back(data_->training_data_[exb + batch*batch_size][0]);
				// calculated weighted inputs and activations
				for (int lyr = 0; lyr != layers_ - 1; ++lyr) {
					w_inputs[exb].push_back( 
						weights_[exb]*(activations[exb][lyr]) + biases_[lyr]
					);
					activations[exb].push_back(sigmoid(w_inputs[exb][lyr]));
				}
				// determine deltas with backpropagation
				delta.push_back(
					backProp(w_inputs[exb], activations[exb], 
					data_->training_data_[exb + batch*batch_size][1])
				);
			}
			// update weights and biases using (ch 1, 20), (ch 1, 21),
			// (ch 2, BP3), (ch2, BP4)
			for (int lyr = 0; lyr != layers_ - 1; ++lyr) {
				Matrix weight_summand = Matrix::Zero(delta[0][lyr].size(),
					activations[0][lyr].size());
				Vector bias_summand = 
					Vector::Zero(delta[0][lyr].size());
				for (int exb = 0; exb != batch_size; ++exb) {
					weight_summand += 
						delta[exb][lyr] * activations[exb][lyr-1].transpose();
					bias_summand += delta[exb][lyr];
				}
				double step = eta / batch_size;
				weights_[lyr] = weights_[lyr] - step*weight_summand;
				biases_[lyr] = biases_[lyr] - step*bias_summand; 
			}
		}
	}
}

std::vector<Vector> ForwardNetwork::backProp(const std::vector<Vector>& w_inputs,
		const std::vector<Vector>& activations, const Vector& output) const {
	std::vector<Vector> delta(layers_ - 1);
	delta[layers_ - 2] = 
		cost_->delta_output(activations[layers_-1],
			w_inputs[layers_-1],
			output
		);
	for (int lyr = layers_ - 3; lyr != -1; --lyr)
		delta[lyr] = coeffProduct(
			weights_[lyr+2].transpose() * delta[lyr+1], 
			sigmoidPrime(w_inputs[lyr])	// (ch2, BP2)
		);
	return std::move(delta);
}



