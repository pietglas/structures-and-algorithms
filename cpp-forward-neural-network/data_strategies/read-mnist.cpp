#include "read-mnist.h"
#include <iostream>
#include <functional>
#include <iterator>

void ReadMNist::read(const std::string& file_path, bool training) {}

void ReadMNist::readData(bool training) {
	auto data = std::ref(training_data_);
	if (!training) 
		data = std::ref(test_data_);
	std::vector<unsigned char> images = readImage(training);
	std::vector<unsigned char> labels = readLabel(training);
	data.get().reserve(labels.size());
	int image_size = images.size() / labels.size();
	for (int i = 0; i < labels.size(); ++i) {
		// convert chars to Eigen Vectors
		Vector label = Vector::Zero(10);
		label((int)labels[i]) = 1;
		Vector image(image_size);
		for (int j = 0; j < image_size; ++j)
			image(j) =  ((double)(images[j + i*image_size])) / 255;
		data.get().emplace_back(image, label);
	}
}

std::vector<unsigned char> ReadMNist::readLabel(bool training) const {
	std::string label_file{"train-labels-idx1-ubyte"};
	if (!training) {
		label_file = std::string("t10k-labels-idx1-ubyte");
	}
	std::ifstream labels(label_file, std::ios::binary);
	if (!labels) {
		std::cerr << "Couln't load label file" << std::endl;
		return std::vector<unsigned char>();
	}
	int lab_magic_nr, nr_labels; 
	readAndReverse(lab_magic_nr, labels);
	if (lab_magic_nr != 2049) {
		std::cerr << "This is not the label set!" << std::endl;
		return std::vector<unsigned char>();
	}
	readAndReverse(nr_labels, labels);
	std::vector<unsigned char> stored_labels;
	stored_labels.reserve(nr_labels);
	// read the data from the file
	stored_labels = std::vector<unsigned char>(
		(std::istreambuf_iterator<char>(labels)),
		(std::istreambuf_iterator<char>())
	);
	return stored_labels;
}

std::vector<unsigned char> ReadMNist::readImage(bool training) const {
	std::string image_file{"train-images-idx3-ubyte"};
	if (!training)
		image_file = std::string("t10k-images-idx3-ubyte");
	// load the data
	std::ifstream images(image_file, std::ios::binary);
	if (!images) {
		std::cerr << "Couldn't load image file" << std::endl;
		return std::vector<unsigned char>();
	}
	int im_magic_nr, nr_images, rows, cols;
	readAndReverse(im_magic_nr, images);
	if (im_magic_nr != 2051) {
		std::cerr << "This is not the image set!" << std::endl;
		return std::vector<unsigned char>();
	}
	readAndReverse(nr_images, images);
	readAndReverse(rows, images);
	readAndReverse(cols, images);
	int image_size = rows * cols;
	std::vector<unsigned char> stored_images;
	stored_images.reserve(nr_images * image_size);
	// read the data from the file
	stored_images = std::vector<unsigned char>(
		(std::istreambuf_iterator<char>(images)),
		(std::istreambuf_iterator<char>())
	);
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

