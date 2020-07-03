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
	Vector vec2 = vec1.unaryExpr([](double x){return exp(x);});
	std::cout << vec1 << std::endl;
	std::cout << vec2 << std::endl;

}