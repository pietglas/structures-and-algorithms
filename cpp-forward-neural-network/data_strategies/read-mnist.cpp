#include "read-mnist.h"
#include <iostream>
#include <functional>

void ReadMNist::read(const std::string& file_path, bool training) {}

void ReadMNist::readData(bool training) {
	auto data = std::ref(training_data_);
	std::string image_file{"train-images-idx3-ubyte"};
	std::string label_file{"train-labels-idx1-ubyte"};
	if (!training) {
		image_file = std::string("t10k-images-idx3-ubyte");
		label_file = std::string("t10k-labels-idx1-ubyte");
		data = std::ref(test_data_);
	}
	// load the data
	std::ifstream images(image_file, std::ios::binary);
	if (!images) {
		std::cerr << "Couldn't load image file" << std::endl;
		return;
	}
	std::ifstream labels(label_file, std::ios::binary);
	if (!labels) {
		std::cerr << "Couln't load label file" << std::endl;
		return;
	}
	int im_magic_nr, nr_images, rows, cols;
	int lab_magic_nr, nr_labels; 
	readAndReverse(im_magic_nr, images);
	readAndReverse(lab_magic_nr, labels);
	if (!(im_magic_nr == 2051 && lab_magic_nr == 2049)) {
		std::cerr << "wrong data set" << std::endl;
		return;
	}
	readAndReverse(nr_images, images);
	readAndReverse(nr_labels, labels);
	if (nr_images != nr_labels) {
		std::cerr << "sets images and labels not compatibel" << std::endl;
		return;
	}
	readAndReverse(rows, images);
	readAndReverse(cols, images);
	int image_size = rows * cols;
	data.get().reserve(nr_images);	// reserve memory to prevent unnecesary resizing
	for (int i = 0; i != nr_images; ++i) {
		Vector image(image_size);
		Vector label = Vector::Zero(10);
		// read an image
		unsigned char pixel;
		for (int j = 0; j != image_size; ++j) {
			images >> pixel;
			image(j) = ((int)pixel) / 255;
		}
		// read the corresponding label
		unsigned char nr;
		labels >> nr;
		// set label at index nr equal to 1 
		if (nr < 10) label((int)nr) = 1; 
		// add the example to the stored data
		data.get().push_back(std::array{image, label});
	}
}

int ReadMNist::reverseInt(int i) const {
	unsigned char c1, c2, c3, c4;
	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;

	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void ReadMNist::readAndReverse(int& i, std::ifstream& stream) const {
	stream.read((char*)&i, sizeof(i));
	i = reverseInt(i);
}

// images.read((char*)&nr_images, sizeof(nr_images));
// 	nr_images = reverseInt(nr_images);
// 	images.read((char*)&rows, sizeof(rows));
// 	rows = reverseInt(rows);
// 	images.read((char*)&cols, sizeof(cols));
// 	cols = reverseInt(cols);
// 	labels.read((char*)&nr_labels, sizeof(nr_labels));
// 	nr_labels = reverseInt(nr_labels);
