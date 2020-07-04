// TODO: fix race condition in last for-loop SGD

#include "forward-network.h"
#include <algorithm>	// for std::random_shuffle
#include <stdexcept>
#include <iostream>
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
	switch (cost_type) {
		case crossentropy:
			cost_= std::make_unique<CrossEntropyCost>();
			break;
		default:
			cost_= std::make_unique<QuadraticCost>();
	}
	// choose a read strategy
	switch (data_type) {
		case binary:
			data_ = std::make_unique<ReadMNist>();
			break;
		default:
			data_ = std::make_unique<ReadText>();
	}
}

void ForwardNetwork::dataSource(const std::string& file, 
		bool training) const {
	if (!file.empty()) 
		data_->read(file, training);	// read training data
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

#pragma omp declare reduction (+: Eigen::MatrixXd: omp_out=omp_out+omp_in)\
	initializer(omp_priv=Eigen::MatrixXd::Zero(omp_orig.rows(), omp_orig.cols()))
#pragma omp declare reduction (+: Eigen::VectorXd: omp_out=omp_out+omp_in)\
	initializer(omp_priv=Eigen::VectorXd::Zero(omp_orig.size()))

void ForwardNetwork::SGD(int epochs, int batch_size, double eta, bool test) {
	// omp_set_num_threads(4);
	for (int epoch = 0; epoch != epochs; ++epoch) {
		std::cout << "current epoch: " << epoch << std::endl;
		// shuffle the data
		std::random_shuffle(data_->training_data_.begin(), 
			data_->training_data_.end());
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
		if (test)
			this->test(false);	// test on training data
	}
}

// auto start = std::chrono::high_resolution_clock::now();
// auto finish = std::chrono::high_resolution_clock::now();
// std::chrono::duration<double> elapsed = finish - start;
// std::cout << "elapsed time: " << elapsed.count() << std::endl;



double ForwardNetwork::test(bool test_data) const {
	double total_cost = 0;
	int size = trainingSize();
	if (test_data)
		size = testSize();
	std::vector<std::vector<Vector>> activations(size);
	std::vector<std::vector<Vector>> w_inputs(size);
	//omp_set_num_threads(4);
	#pragma omp parallel for default(none) \
		shared(activations, w_inputs, size, total_cost, test_data) \
		schedule(guided)
	for (int ex = 0; ex < size; ++ex) {
		activations[ex].reserve(layers_);
		w_inputs[ex].reserve(layers_ - 1);
		feedForward(activations[ex], w_inputs[ex], ex);
		// add the cost for this example to the total cost
		if (test_data) 
			total_cost += cost_->costFunction(
				activations[ex][layers_-1], data_->test_data_[ex].second
			);
		else 	// test on training data if test data is not available
			total_cost += cost_->costFunction(
				activations[ex][layers_-1], data_->training_data_[ex].second
			);
	}
	std::cout << "cost for this batch: " << total_cost / size 
		<< std::endl;
	return total_cost / size;
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
		std::vector<Vector>& delta, const Vector& output) const {
	delta.resize(layers_ - 1);
	delta[layers_ - 2] = 
		cost_->deltaOutput(activations[layers_-1], output,
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
		std::default_random_engine generator(rd());
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



