#include "forward-network.h"
#include <vector>
#include <iostream>
#include <chrono>

int main(int argc, char **argv) {
	auto start = std::chrono::high_resolution_clock::now();
	std::vector sizes{784, 15, 10};
	ForwardNetwork simple_nn{sizes, quadratic, binary};
	std::string data;
	if (argc == 2) {
		data = std::string{argv[1]};
	}
	//std::cout << "eta this run: " << eta << std::endl;
	simple_nn.dataSource(data, true);
	simple_nn.SGD(1, 20, 3, true);
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "elapsed time: " << elapsed.count() << std::endl;
	simple_nn.resetNetwork();

}