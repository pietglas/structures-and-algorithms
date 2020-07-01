#include "forward-network.h"
#include <vector>
#include <iostream>

int main(int argc, char **argv) {
	std::vector sizes{784, 15, 10};
	ForwardNetwork simple_nn{sizes, quadratic, binary};
	std::string data;
	if (argc == 2) {
		data = std::string{argv[1]};
	}
	//std::cout << "eta this run: " << eta << std::endl;
	simple_nn.dataSource(data, true);
	simple_nn.SGD(30, 10, 3, true);
	simple_nn.resetNetwork();
}