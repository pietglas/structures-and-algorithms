#include <string>
#include <array>
#include <vector>
#include "read-data.h"
#include <eigen/Eigen/Dense>

using Vector = Eigen::VectorXd;

class ReadMNist : public ReadData {
public:
	virtual void read(const std::string& file_path, bool training) override;
	virtual void readData(const std::string& file_path, bool training) override;
	virtual void readLabel(const std::string& file_path, bool training) override;
};