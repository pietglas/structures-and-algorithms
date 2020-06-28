#include "forward-network.h"
#include <algorithm>	// for std::random_shuffle

using Matrix = Eigen::MatrixXd;
using std::unique_ptr;
using std::shared_ptr;
using Vector = Eigen::VectorXd;

ForwardNetworK::ForwardNetwork(std::vector<int> layer_sizes, 
		CostFunction cost_type) : layers_{layer_sizes.size()},
		layer_sizes_{layer_sizes} {
	for (int i = 0; i != layer_sizes - 2; ++i) {
		biases_[i] = Vector::Random(layer_sizes[i]);
		weights_[i-1] = Matrix::Random(layer_sizes[i], layer_sizes[i-1]);
	}
	switch (cost_type) {
		case crossentropy:
			cost_= make_unique<CrossEntropyCost>();
			break;
		default:
			cost_= make_unique<QuadraticCost>();
	}
}

void ForwardNetworK::SGD(std::vector<std::array<Vector, 2>>& training_data, 
		int epochs, int batch_size, double eta) {
	for (int epoch = 0; epoch != epochs; ++epoch) {
		// shuffle the data
		std::random_shuffle(training_data.begin(), training_data.end());
		// for every learning step, we only consider a subset of the data
		const int nr_of_batches = training_data.size() / batch_size;
		int ex = 0;	// to keep track of the current training example
		for (int batch_nr = 0; batch_nr != nr_of_batches; ++batch) {
			// for every training example in the base, we have a vector
			// of Vectors, where every Vector contains a layer of 
			// activations/weighted inputs/deltas, respectively
			std::vector<std::vector<Vector>> activations(batch_size);
			std::vector<std::vector<Vector>> weighted_inputs(batch_size)
			std::vector<std::vector<Vector>> delta;
			delta.reserve(batch_size);
			// for each training example, apply backpropagation 
			for (int exb = 0; exb != batch_size; ++exb) {
				// set training example input
				activations[exb][0] = training_data[ex][0];
				// calculated weighted inputs and activations
				for (int lyr = 0; lyr != layers_ - 1; ++lyr) {
					weighted_inputs[exb][lyr] = 
						weights_[exb]*(activations[exb][lyr]) + biases_[lyr];
					activations[exb][lyr+1] = sigmoid(weighted_inputs[exb][lyr]);
				}
				// determine deltas with backpropagation
				delta.push_back(
					backProp(weighted_inputs[exb], activations[exb], ex)
				);
				++ex;
			}
			// update weights and biases using (ch 1, 20), (ch 1, 21),
			// (ch 2, BP3), (ch2, BP4)
			for (int lyr = 0; lyr != layers_ - 1; ++lyr) {
				Matrix update_weight_summand;
				Vector update_bias_summand;
				for (int exb = 0; exb != batch_size; ++exb) {
					update_weight_summand += 
						delta[exb][lyr] * activations[exb][lyr-1].transpose();
					update_bias_summand += delta[exb][lyr];
				}
				double step = eta / batch_size;
				weights_[lyr] = weights_[lyr] - step*update_weight_summand;
				biases_[lyr] = biases_[lyr] - step*update_bias_summand; 
			}
		}
	}
}

std::vector<Vector> ForwardNetwork::backProp(const std::vector<Vector>& w_inputs,
		const std::vector<Vector>& activations, int ex) const {
	std::vector<Vector> delta(layers_ - 1);
	delta[layers_ - 2] = 
		cost_->delta_output(activations[layers_-1],
			w_inputs[layers_-1],
			training_data[ex][1]
		);
	for (int lyr = layers_ - 3; lyr != -1; --lyr)
		delta[lyr] = coeffProduct(
			weights_[lyr+2].transpose() * delta[lyr+1], 
			sigmoidPrime(w_inputs[lyr])	// (ch2, BP2)
		);
	return std::move(delta);
}

