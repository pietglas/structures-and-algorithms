#include "read-text.h"

void ReadText::read(const std::string& file_path, bool training) {
	int input_size;
	int output_size;
	int size;
	ifstream data(file_path);
	if (!data)
		cout << "Could not load the data" << endl;
	data >> input_size >> output_size;
	if (training) {
		data >> train_size_;
		size = train_size_;
	}
	else {
		data >> test_size_;
		size = test_size_;
	}
	training_data_.reserve(size);
	for (int ex = 0; ex != size; ++ex) {
		Vector input(input_size);
		Vector output(output_size);
		// get input neurons from file 
		for (int i = 0; i != input_size; i++)
			data >> input(i);
		// get output neurons from file
		for (int j = 0; j != output_size; j++)
			data >> output(j);
		if (training)
			training_data_.push_back(std::array{input, output});
		else
			test_data_.push_back(std::array{input, output});
	}
}

void ReadText::readData(const std::string& file_path, bool training) {}
void ReadText::readLabel(const std::string& file_path, bool training) {}
