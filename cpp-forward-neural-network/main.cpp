#include "forward-network.h"
#include <vector>
#include <iostream>
#include <chrono>
#include <omp.h>

int main(int argc, char **argv) {
	auto start = std::chrono::high_resolution_clock::now();
	omp_set_num_threads(4);
	std::vector sizes{784, 15, 10};
	ForwardNetwork simple_nn{sizes, quadratic, binary};
	std::string data;
	if (argc == 2) {
		data = std::string{argv[1]};
	}
	//std::cout << "eta this run: " << eta << std::endl;
	simple_nn.dataSource(data, true);
	for (int i = 1; i != 6; ++i) {
		std::cout << "eta this run: " << i*0.4 << std::endl;
		simple_nn.SGD(10, 40, 0.4*i, true);
		simple_nn.resetNetwork();
	}
	
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "elapsed time: " << elapsed.count() << std::endl;
}