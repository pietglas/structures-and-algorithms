#pragma once
#include <string>
#include "read-data.h"
#include <eigen/Eigen/Dense>
#include <fstream>
#include <iostream>
#include <array>
#include <vector>

using std::ifstream;
using std::cout;
using std::endl;
using Vector = Eigen::VectorXd;

class ReadText : public ReadData {
public:
	virtual void read(const std::string& file_path, bool training) override;
	virtual void readData(const std::string& file_path, bool training) override;
	virtual void readLabel(const std::string& file_path, bool training) override;
};


