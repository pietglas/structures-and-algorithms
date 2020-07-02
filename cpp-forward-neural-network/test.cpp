// Example program
#include <iostream>
#include <string>
#include <chrono>
#include <vector>
#include <eigen/Eigen/Dense>
#include <omp.h>

using Vector = Eigen::VectorXd;

int main()
{
	auto start = std::chrono::high_resolution_clock::now();
	Vector vec = Vector::Zero(100);
	Vector vec_ = Vector::Ones(100);
	std::vector<Vector> vecvec(1000000);
	omp_set_num_threads(3);
	#pragma omp parallel for
	for (int i = 0; i < 1000000; ++i) {
		vecvec[i] = vec + vec_;
	}

	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "elapsed time: " << elapsed.count() << std::endl;

	start = std::chrono::high_resolution_clock::now();
	std::vector<Vector> vecvec_(1000000);
	for (int i = 0; i < 1000000; ++i) {
		vecvec_[i] = vec_ + vec;
	}
	finish = std::chrono::high_resolution_clock::now();
	elapsed = finish - start;
	std::cout << "elapsed time: " << elapsed.count() << std::endl;
}