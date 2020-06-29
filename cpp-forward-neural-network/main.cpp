#include "forward-network.h"
#include <vector>
#include <iostream>

int main(int argc, char **argv) {
	std::vector sizes{1, 1};
	ForwardNetwork simple_nn{sizes, quadratic, text};
	if (argc == 1) {
		std::cerr << "no data provided" << std::endl;
		return -1;
	}
	std::vector data{std::string(argv[1])};
	simple_nn.dataSource(data, true);
	simple_nn.SGD(1, simple_nn.trainingSize());
}