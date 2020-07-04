// Example program
#include <iostream>
#include <string>
#include <chrono>
#include <vector>
#include <cmath>
#include <eigen/Eigen/Dense>
#include <omp.h>

using Vector = Eigen::VectorXd;


int main()
{
	Vector vec1 = Vector::Ones(10);
	vec1(4) = 2;
	Vector::Index max_index;
	double max_val = vec1.rowwise().sum().maxCoeff(&max_index);
	std::cout << "Max value at index " << (int)max_index << std::endl;
}