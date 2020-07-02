#pragma once
#include <string>
#include <vector>
#include <utility>
#include <eigen/Eigen/Dense>

class ForwardNetwork;

using Vector = Eigen::VectorXd;

class ReadData {
	friend class ForwardNetwork;
public:
	virtual void read(const std::string& file_path, bool training) = 0;
	virtual void readData(bool training) = 0;
protected:
	std::vector<std::pair<Vector, Vector>> training_data_;
	std::vector<std::pair<Vector, Vector>> test_data_;
	int train_size_;
	int test_size_;
};