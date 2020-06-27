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
			cost_function_ = make_unique<CrossEntropyCost>();
			break;
		default:
			cost_function_ = make_unique<QuadraticCost>();
	}
}

void ForwardNetworK::SGD(std::vector<std::array<Vector, 2>>& training_data, 
		int epochs, int batch_size, double eta) {
	for (int epoch = 0; epoch != epochs; ++epoch) {
		// shuffle the data
		std::random_shuffle(training_data.begin(), training_data.end());
		// for every learning step, we only consider a subset of the data
		const int nr_of_batches = training_data.size() / batch_size;
		int training_example = 0;
		for (int batch_nr = 0; batch_nr != nr_of_batches; ++batch) {
			std::vector<std::vector<Vector>> activations(batch_size);
			std::vector<std::vector<Vector>> weighted_inputs(batch_size)
			std::vector<std::vector<Vector>> delta(batch_size);
			// for each training example, apply backpropagation 
			for (int input_nr = 0; input_nr != batch_size; ++input_nr) {
				// set training example input
				activations[input_nr][0] = training_data[input_nr][0];
				// calculated weighted inputs and activations
				for (int layer = 1; layer != layers_; ++layer) {
					weighted_inputs[input_nr][layer-1] = 
						weights_[input_nr]*(activations[input_nr][layer]) 
						+ biases_[i];
					activations[input_nr][layer-1] = 
						sigmoid(weighted_inputs[input_nr][layer-1]);
				}
			}
			
		}
	}
	

	std::vector<Vector> activations(layers_);
	activations[0] = 
}

// returns a vector with the delta vector of each layer
std::vector<Vector> ForwardNetworK::backPropagation(const Vector& input,
	const Vector& output) const {

}
