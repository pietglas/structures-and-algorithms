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
	unsigned char pixel = 244; 
	double d_pixel = ((double)pixel) / 255;
	std::cout << d_pixel << std::endl;
}