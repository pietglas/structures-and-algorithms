#include "read-text.h"
#include <iostream>
#include <functional>

void ReadText::read(const std::string& file_path, bool training) {
	int input_size;
	int output_size;
	int size;
	auto train_or_test = std::ref(training_data_);
	ifstream data(file_path);
	if (!data) {
		cout << "Could not load the data" << endl;
		train_or_test = std::ref(test_data_);
	}
	data >> input_size >> output_size;
	if (training) {
		data >> train_size_;
		size = train_size_;
	}
	else {
		data >> test_size_;
		size = test_size_;
	}
	train_or_test.get().reserve(size);
	for (int ex = 0; ex != size; ++ex) {
		Vector input(input_size);
		Vector output(output_size);
		// get input neurons from file 
		for (int i = 0; i != input_size; i++)
			data >> input(i);
		// get output neurons from file
		for (int j = 0; j != output_size; j++)
			data >> output(j);
		train_or_test.get().push_back(std::array{input, output});
	}
}

void ReadText::readData(bool training) {}

