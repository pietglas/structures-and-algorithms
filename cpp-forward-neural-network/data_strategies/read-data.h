#pragma once
#include <string>
#include <array>
#include <vector>
#include <eigen/Eigen/Dense>

class ForwardNetwork;

using Vector = Eigen::VectorXd;

class ReadData {
	friend class ForwardNetwork;
public:
	virtual void read(const std::string& file_path, bool training) = 0;
	virtual void readData(bool training) = 0;
protected:
	std::vector<std::array<Vector, 2>> training_data_;
	std::vector<std::array<Vector, 2>> test_data_;
	int train_size_;
	int test_size_;
};