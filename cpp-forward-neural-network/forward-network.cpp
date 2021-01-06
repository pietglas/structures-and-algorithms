// TODO: fix race condition in last for-loop SGD

#include "forward-network.h"
#include <algorithm>	// for std::random_shuffle
#include <numeric> // for std::accumulate
#include <stdexcept>
#include <iostream>
#include <functional> // for std::ref
#include <random>
#include <chrono>
//#include <omp.h>

using Matrix = Eigen::MatrixXd;
using std::unique_ptr;
using std::shared_ptr;
using Vector = Eigen::VectorXd;

ForwardNetwork::ForwardNetwork(std::vector<int> layer_sizes, 
		CostFunction cost_type, DataType data_type) : 
		layers_{layer_sizes.size()}, layer_sizes_{layer_sizes} {
	// set weights and biases
	this->setWeightsBiasesRandom();
	// set cost function type 
	this->setCost(cost_type);
	// choose a read strategy
	this->setDataType(data_type);
	
}

void ForwardNetwork::data(bool training, const std::string& file) {
	if (!file.empty()) {
		training ? training_data_ = file : test_data_ = file;
		data_->read(file, training);	// read training data
	}
	else 
		data_->readData(training);
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

void ForwardNetwork::setDataType(const DataType& data_type) {
	switch (data_type) {
		case mnist:
			data_ = std::make_unique<ReadMNist>();
			break;
		default:
			data_ = std::make_unique<ReadText>();
	}
}

void ForwardNetwork::setCost(const CostFunction& cost_function) {
	switch (cost_function) {
		case crossentropy:
			cost_= std::make_unique<CrossEntropyCost>();
			break;
		default:
			cost_= std::make_unique<QuadraticCost>();
	}
}

// user defined reductions for Eigen vectors and matrices; prevents data
// races when multithreading a for-loop in which a shared variable is 
// modified
// #pragma omp declare reduction (+: Eigen::MatrixXd: omp_out=omp_out+omp_in)\
// 	initializer(omp_priv=Eigen::MatrixXd::Zero(omp_orig.rows(), omp_orig.cols()))
// #pragma omp declare reduction (+: Eigen::VectorXd: omp_out=omp_out+omp_in)\
// 	initializer(omp_priv=Eigen::VectorXd::Zero(omp_orig.size()))

void ForwardNetwork::SGD(int epochs, int batch_size, double eta, bool test, bool testdata) {
	auto& data = data_->training_data_;
	std::random_device rd;
	std::default_random_engine rng{rd()};
	for (int epoch = 0; epoch != epochs; ++epoch) {
		// shuffle the data
		std::shuffle(std::begin(data), std::end(data), rng);
		// divide the training data in batches of size batch_size
		int nr_batches = data.size() / batch_size;
		for (int batch = 0; batch != nr_batches; ++batch) {
			// for every training example in the batch, we have a vector
			// of Vectors, where every Vector contains a layer of 
			// activations/weighted inputs/deltas, respectively			
			std::vector<std::vector<Vector>> nabla_b(layers_-1);
			std::vector<std::vector<Matrix>> nabla_w(layers_-1);
			for (int lyr = 0; lyr < layers_-1; ++lyr) {
				nabla_b[lyr].resize(batch_size);
				nabla_w[lyr].resize(batch_size);
			}
			// for each training example, apply backpropagation. 
			// parallelizing gives slight performance increase
			#pragma omp parallel for default(none) \
				shared(nabla_w, nabla_b, batch_size, batch) \
				schedule(guided)

			for (int exb = 0; exb < batch_size; ++exb) {
				std::vector<Vector> activations;
				std::vector<Vector> w_inputs;
				int current_ex = exb + batch_size*batch;
				// feedforward to calculate activations and weighted inputs
				feedForward(activations, w_inputs, current_ex);
				// determine deltas with backpropagation
				backProp(activations, w_inputs, nabla_b,  
					nabla_w, exb, current_ex);
			}			
			// update weights and biases using (ch 1, 20), (ch 1, 21),
			// (ch 2, BP3), (ch2, BP4)
			double stepsize = eta / batch_size;
			for (int lyr = weights_.size() - 1; lyr > -1; --lyr) {
				Vector zerovec = Vector::Zero(biases_[lyr].size());
				Matrix zeromat = 
					Matrix::Zero(weights_[lyr].rows(), weights_[lyr].cols());
				Vector update_bias = 
					std::accumulate(nabla_b[lyr].begin(), nabla_b[lyr].end(),
					zerovec);
				Matrix update_weight = 
					std::accumulate(nabla_w[lyr].begin(), nabla_w[lyr].end(),
					zeromat);
				biases_[lyr].noalias() -= stepsize * update_bias; 
				weights_[lyr].noalias() -= stepsize * update_weight;
			}
		}
		if (test)
			this->test(testdata);	// test 
	}
}

void ForwardNetwork::test(bool test_data) {
	int correct_examples = 0;
	auto data = std::ref(data_->training_data_);
	int size = trainingSize();
	if (test_data) {
		if (data_->test_data_.empty())
			this->data(false, test_data_);
		size = testSize();
		data = std::ref(data_->test_data_);
	}

	#pragma omp parallel for default(shared) \
		reduction(+: correct_examples)

	for (int ex = 0; ex < size; ++ex) {
		std::vector<Vector> activations;
		std::vector<Vector> w_inputs;

		feedForward(activations, w_inputs, ex);
		// output is correct when the largest value has the same index
		// as the index of the expected output with value 1
		Vector::Index max_index_exp;
		Vector::Index max_index_out;
		
		data.get()[ex].second.rowwise().sum().maxCoeff(&max_index_exp);
		activations[layers_-1].rowwise().sum().maxCoeff(&max_index_out);

		if (max_index_out == max_index_exp) {
			++correct_examples;
		}
	}
	std::cout << "number of correct examples: " << correct_examples <<
		" / " << size << std::endl;
}

void ForwardNetwork::resetNetwork() {
	biases_ = std::vector<Vector>();
	weights_ = std::vector<Matrix>();
	setWeightsBiasesRandom();
}

void ForwardNetwork::feedForward(std::vector<Vector>& activations,
		std::vector<Vector>& w_inputs, int train_ex) const {
	// activate first layer
	activations.emplace_back(data_->training_data_[train_ex].first);
	// calculated weighted inputs and activations
	for (int lyr = 0; lyr != layers_ - 1; ++lyr) {
		w_inputs.emplace_back( 
			weights_[lyr]*(activations[lyr]) + biases_[lyr]
		);
		activations.emplace_back(sigmoid(w_inputs[lyr]));
	}
}

void ForwardNetwork::backProp(const std::vector<Vector>& activations, 
		const std::vector<Vector>& w_inputs, 
		std::vector<std::vector<Vector>>& nabla_b,
		std::vector<std::vector<Matrix>>& nabla_w, 
		int batch_ex, int train_ex) const {
	auto& data = data_->training_data_;
	Vector delta = 
		cost_->deltaOutput(activations[layers_-1], data[train_ex].second,
			w_inputs[layers_-2]
		);
	nabla_b[layers_-2][batch_ex] = delta;
	nabla_w[layers_-2][batch_ex] = delta * (activations[layers_-2].transpose());
	for (int lyr = layers_ - 3; lyr > -1; --lyr) {
		delta = 
			(weights_[lyr+1].transpose() * delta).cwiseProduct( 
			sigmoidPrime(w_inputs[lyr])	// (ch2, BP2)
		);
		nabla_b[lyr][batch_ex] = delta;
		nabla_w[lyr][batch_ex] = delta * (activations[lyr].transpose());
	}
}

void ForwardNetwork::setWeightsBiasesRandom() {
	std::random_device rd;
	std::default_random_engine generator{rd()};
	for (int i = 0; i != layers_- 1; ++i) {
		// set gaussian distribution with mean 0 and standard dev 1
		std::normal_distribution<double> distribution{0, 1};
		auto normal = [&] (double) {return distribution(generator);};
		// initialize weights and biases randomly
		biases_.emplace_back(
			Vector::NullaryExpr(layer_sizes_[i+1], normal)
		);
		weights_.emplace_back(
			Matrix::NullaryExpr(layer_sizes_[i+1], layer_sizes_[i], normal)
		);
	}
}



