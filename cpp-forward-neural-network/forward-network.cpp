// TODO: fix race condition in last for-loop SGD

#include "forward-network.h"
#include <algorithm>	// for std::random_shuffle
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
#pragma omp declare reduction (+: Eigen::MatrixXd: omp_out=omp_out+omp_in)\
	initializer(omp_priv=Eigen::MatrixXd::Zero(omp_orig.rows(), omp_orig.cols()))
#pragma omp declare reduction (+: Eigen::VectorXd: omp_out=omp_out+omp_in)\
	initializer(omp_priv=Eigen::VectorXd::Zero(omp_orig.size()))

void ForwardNetwork::SGD(int epochs, int batch_size, double eta, bool test) {
	for (int epoch = 0; epoch != epochs; ++epoch) {
		std::cout << "current epoch: " << epoch << std::endl;
		// std::cout << "bias second layer: " << biases_[0] << std::endl;
		std::cout << "bias third layer: " << biases_[1] << std::endl;
		// std::cout << "weights first layer: " << weights_[0] << std::endl;
		// std::cout << "weights second layer: " << weights_[1] << std::endl;
		// shuffle the data
		std::random_device rd;
		std::default_random_engine rng{rd()};
		std::shuffle(std::begin(data_->training_data_), 
			std::end(data_->training_data_), rng);
		// divide the training data in batches of size batch_size
		int nr_of_batches = data_->training_data_.size() / batch_size;
		for (int batch = 0; batch != nr_of_batches; ++batch) {
			// std::cout << "batch: " << batch << std::endl;
			
			// for every training example in the batch, we have a vector
			// of Vectors, where every Vector contains a layer of 
			// activations/weighted inputs/deltas, respectively
			
			// auto start = std::chrono::high_resolution_clock::now();
			
			std::vector<std::vector<Vector>> activations(batch_size);
			std::vector<std::vector<Vector>> w_inputs(batch_size);
			std::vector<std::vector<Vector>> delta(batch_size);
			// for each training example, apply backpropagation. 
			// parallelizing gives slight performance increase
			#pragma omp parallel for default(none) \
				shared(activations, w_inputs, delta, batch_size, batch) \
				schedule(guided)
			for (int exb = 0; exb < batch_size; ++exb) {
				// feedforward to calculate activations and weighted inputs
				feedForward(activations[exb], w_inputs[exb], exb+batch_size*batch);
				// determine deltas with backpropagation
				backProp(activations[exb], w_inputs[exb],  
					delta[exb], data_->training_data_[exb+batch_size*batch].second);
			}
			// auto finish = std::chrono::high_resolution_clock::now();
			// std::chrono::duration<double> elapsed = finish - start;
			// std::cout << "elapsed time feedforward & backprop: " << 
			// 	elapsed.count() << std::endl;
			
			// update weights and biases using (ch 1, 20), (ch 1, 21),
			// (ch 2, BP3), (ch2, BP4)
			
			// start = std::chrono::high_resolution_clock::now();
			double step = eta / batch_size;
			for (int lyr = 0; lyr < layers_ - 1; ++lyr) {
				Matrix weight_summand = 
					Matrix::Zero(weights_[lyr].rows(), weights_[lyr].cols());
				Vector bias_summand = 
					Vector::Zero(biases_[lyr].size());
				// parallelizing gives performance increase of about a
				// 100% !
				#pragma omp parallel for default(shared) \
					reduction(+: weight_summand, bias_summand)

				for (int exb = 0; exb < batch_size; ++exb) {
					weight_summand.noalias() += 
						delta[exb][lyr] * (activations[exb][lyr].transpose());
					bias_summand.noalias() += delta[exb][lyr];
				}
				weights_[lyr].noalias() -= step*weight_summand;
				biases_[lyr].noalias() -= step*bias_summand; 
			}
			// finish = std::chrono::high_resolution_clock::now();
			// elapsed = finish - start;
			// std::cout << "elapsed time sgd: " << elapsed.count() << std::endl;
		}
		if (test) {
			this->test(true);	// test 
		}
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
		double max_val = 
			data.get()[ex].second.rowwise().sum().maxCoeff(&max_index_exp);
		max_val = 
			activations[layers_-1].rowwise().sum().maxCoeff(&max_index_out);
		if (max_index_out == max_index_exp)
			++correct_examples;
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
	// reserve enough memory to prevent many resize operations
	activations.reserve(layers_);
	w_inputs.reserve(layers_-1);
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
		std::vector<Vector>& delta, const Vector& expected) const {
	delta.resize(layers_ - 1);
	delta[layers_ - 2] = 
		cost_->deltaOutput(activations[layers_-1], expected,
			w_inputs[layers_-2]
		);
	for (int lyr = layers_ - 3; lyr > -1; --lyr)
		delta[lyr] = 
			(weights_[lyr+1].transpose() * delta[lyr+1]).cwiseProduct( 
			sigmoidPrime(w_inputs[lyr])	// (ch2, BP2)
		);
}

void ForwardNetwork::setWeightsBiasesRandom() {
	biases_.reserve(layers_-1);
	weights_.reserve(layers_-1);
	for (int i = 0; i != layers_- 1; ++i) {
		// set gaussian distribution with mean 0 and standard dev 1
		std::random_device rd;
		std::default_random_engine generator{rd()};
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



