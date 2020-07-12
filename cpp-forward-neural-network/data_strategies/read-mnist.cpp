#include "read-mnist.h"
#include <iostream>
#include <functional>

void ReadMNist::read(const std::string& file_path, bool training) {}

void ReadMNist::readData(bool training) {
	auto data = std::ref(training_data_);
	// std::string image_file{"train-images-idx3-ubyte"};
	// std::string label_file{"train-labels-idx1-ubyte"};
	if (!training) {
		// image_file = std::string("t10k-images-idx3-ubyte");
		// label_file = std::string("t10k-labels-idx1-ubyte");
		data = std::ref(test_data_);
	}
	// load the data
	// std::ifstream images(image_file, std::ios::binary);
	// if (!images) {
	// 	std::cerr << "Couldn't load image file" << std::endl;
	// 	return;
	// }
	// std::ifstream labels(label_file, std::ios::binary);
	// if (!labels) {
	// 	std::cerr << "Couln't load label file" << std::endl;
	// 	return;
	// }
	// int im_magic_nr, nr_images, rows, cols;
	// int lab_magic_nr, nr_labels; 
	// readAndReverse(im_magic_nr, images);
	// readAndReverse(lab_magic_nr, labels);
	// if (!(im_magic_nr == 2051 && lab_magic_nr == 2049)) {
	// 	std::cerr << "wrong data set" << std::endl;
	// 	return;
	// }
	// readAndReverse(nr_images, images);
	// readAndReverse(nr_labels, labels);
	// readAndReverse(rows, images);
	// readAndReverse(cols, images);
	// int image_size = rows * cols;
	// data.get().reserve(nr_images);	// reserve memory to prevent unnecesary resizing
	// for (int i = 0; i < nr_images; ++i) {
	// 	Vector image(image_size);
	// 	Vector label = Vector::Zero(10);
	// 	// read an image
	// 	unsigned char pixel;
	// 	for (int j = 0; j != image_size; ++j) {
	// 		images >> pixel;
	// 		image(j) = /*((double)pixel) / 255*/pixel;
	// 	}
	// 	// read the corresponding label
	// 	unsigned char nr;
	// 	labels >> nr;
	// 	int index = nr;
	// 	if (i < 55100 && i > 55000) {
	// 		std::cerr << "index: " << index << std::endl;
	// 		for (int j = 1; j < image_size + 1; ++j) {
	// 			std::cerr << image(j-1) << " ";
	// 			if ((j % 28) == 0 && j != 0)
	// 				std::cerr << std::endl;
	// 		}
	// 	}
	// 	// set label at index nr equal to 1 
	// 	// nr < 10 ? label(index) = 1 : label(index - 1) = 1; 
	// 	if (index < 10) label(index) = 1;
		// add the example to the stored data
	// 	data.get().emplace_back(image, label);
	// }
	std::vector<Vector> images = readImage(training);
	std::vector<Vector> labels = readLabel(training);
	for (int i = 0; i < images.size(); i++) {
		data.get().emplace_back(images[i], labels[i]);
	}
}

std::vector<Vector> ReadMNist::readLabel(bool training) const {
	std::string label_file{"train-labels-idx1-ubyte"};
	if (!training) {
		label_file = std::string("t10k-labels-idx1-ubyte");
	}
	std::ifstream labels(label_file, std::ios::binary);
	if (!labels) {
		std::cerr << "Couln't load label file" << std::endl;
		return std::vector<Vector>();
	}
	int lab_magic_nr, nr_labels; 
	readAndReverse(lab_magic_nr, labels);
	if (lab_magic_nr != 2049) {
		std::cerr << "wrong data set" << std::endl;
		return std::vector<Vector>();
	}
	readAndReverse(nr_labels, labels);
	std::vector<Vector> stored_labels;
	stored_labels.reserve(nr_labels);
	for (int i = 0; i < nr_labels; ++i) {
		Vector label = Vector::Zero(10);
		// read the corresponding label
		unsigned char nr;
		labels >> nr;
		int index = nr;
		if (index < 10) label(index) = 1;
		// add the example to the stored data
		stored_labels.emplace_back(label);
	}
	return stored_labels;
}

std::vector<Vector> ReadMNist::readImage(bool training) const {
	std::string image_file{"train-images-idx3-ubyte"};
	if (!training) {
		image_file = std::string("t10k-images-idx3-ubyte");
	}
	// load the data
	std::ifstream images(image_file, std::ios::binary);
	if (!images) {
		std::cerr << "Couldn't load image file" << std::endl;
		return std::vector<Vector>();
	}
	int im_magic_nr, nr_images, rows, cols;
	readAndReverse(im_magic_nr, images);
	if (im_magic_nr != 2051) {
		std::cerr << "wrong data set" << std::endl;
		return std::vector<Vector>();
	}
	readAndReverse(nr_images, images);
	readAndReverse(rows, images);
	readAndReverse(cols, images);
	int image_size = rows * cols;
	std::vector<Vector> stored_images;
	stored_images.reserve(nr_images);
	for (int i = 0; i < nr_images; ++i) {
		Vector image(image_size);
		// read an image
		unsigned char pixel;
		for (int j = 0; j < image_size; ++j) {
			images >> pixel;
			image(j) = /*((double)pixel) / 255*/pixel;
		}
		stored_images.emplace_back(image);
	}
	return stored_images;
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

