#include "forward-network.h"
#include <vector>
#include <iostream>

int main(int argc, char **argv) {
	std::vector sizes{5, 10, 1};
	ForwardNetwork simple_nn{sizes, quadratic, text};
	if (argc == 1) {
		std::cerr << "no data provided" << std::endl;
		return -1;
	}
	std::vector data{std::string(argv[1])};
	simple_nn.dataSource(data, true);
	for (int i = 1; i != 11; i++) {
		double eta = i*0.2;
		std::cout << "eta this run: " << eta << std::endl;
		simple_nn.SGD(100, simple_nn.trainingSize(), eta, true);
		simple_nn.resetNetwork();
	}
}