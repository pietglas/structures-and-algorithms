#include "forward-network.h"
#include <vector>

int main(int argc, char **argv) {
	std::vector sizes{1, 1};
	ForwardNetwork simple_nn{sizes, quadratic, text};
	std::vector data{std::string(argv[1])};
	simple_nn.dataSource(data, true);
	simple_nn.SGD(1, simple_nn.trainingSize());
}