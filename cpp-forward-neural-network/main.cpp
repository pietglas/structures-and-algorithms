#include "forward-network.h"
#include <vector>
#include <iostream>
#include <chrono>
#include <omp.h>

int main(int argc, char **argv) {
	auto start = std::chrono::high_resolution_clock::now();
	// omp_set_num_threads(4);
	std::vector sizes{784, 30, 10};
	ForwardNetwork simple_nn{sizes, quadratic, mnist};
	std::string data;
	if (argc == 2) {
		data = std::string{argv[1]};
	}
	simple_nn.data(true, data);	// read train data
	simple_nn.data(false, data);	// read test data
	simple_nn.SGD(2, 20, 3, true);
	simple_nn.resetNetwork();
	
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "elapsed time: " << elapsed.count() << std::endl;
}