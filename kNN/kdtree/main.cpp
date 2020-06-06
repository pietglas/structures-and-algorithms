#include "kdtree.hpp"
#include <iostream>

int main() {
	KDTree<2, std::string> twodtree;
	std::array<double, 2> twodcoords1 = {0, 0};
	std::array<double, 2> twodcoords2 = {-1, 1};
	std::array<double, 2> twodcoords3 = {1, 0};
	std::string b1 = "building1";
	std::string b2 = "building2";
	std::string b3 = "building3";

	twodtree.add(twodcoords1, b1);
	twodtree.add(twodcoords2, b2);
	twodtree.add(twodcoords3, b3);

	twodtree.print();

	KDTree<2, std::string> twodtree2 = twodtree;

	twodtree2.print();

	std::array<double, 2> point0 = {-2, 1};
	std::array<double, 2> point1 = {2, 0};

	std::vector<std::string> knn0 = twodtree.kNN(point0, "building4", 1);
	std::vector<std::string> knn1 = twodtree.kNN(point1, "building5", 2);

	for (auto i : knn0)
		std::cout << "element " << i << ": " << i << std::endl;
	for (auto i : knn1)
		std::cout << "element " << i << ": " << i << std::endl;
}