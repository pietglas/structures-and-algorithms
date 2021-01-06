#include "forward-network.h"
#include <vector>
#include <iostream>
#include <chrono>
#include <omp.h>

int main(int argc, char **argv) {
	auto start = std::chrono::high_resolution_clock::now();
	omp_set_num_threads(4);
	std::vector sizes{784, 100, 10};
	ForwardNetwork simple_nn{sizes, quadratic, mnist};
	std::string data;
	bool test = true;
	bool testdata = true;
	if (argc == 3) {
		//data = std::string{argv[1]};
		std::string argv1 = argv[1];
		std::string argv2 = argv[2];
		test = argv1 == "true" ? true : false;
		testdata = argv2 == "true" ? true : false;
	}
	simple_nn.data(true, data);	// load training data
	simple_nn.data(false, data);	// load test data
	simple_nn.SGD(10, 20, 0.1, test, testdata);
	simple_nn.resetNetwork();
	
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "elapsed time: " << elapsed.count() << std::endl;
}