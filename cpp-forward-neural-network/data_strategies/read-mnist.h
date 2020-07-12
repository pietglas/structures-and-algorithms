#include <string>
#include <array>
#include <vector>
#include <fstream>
#include "read-data.h"
#include <eigen/Eigen/Dense>

using Vector = Eigen::VectorXd;

class ReadMNist : public ReadData {
public:
	virtual void read(const std::string& file_path, bool training) override;
	virtual void readData(bool training) override;
private:
	int reverseInt(int i) const;
	void readAndReverse(int& i, std::ifstream& stream) const;
	std::vector<Vector> readLabel(bool training) const;
	std::vector<Vector> readImage(bool training) const;
};