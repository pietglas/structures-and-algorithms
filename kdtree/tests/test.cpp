#include "../kdtree.hpp"
#include <string>
#include <array>
#include <gtest/gtest.h>

class KDTreeTest : public ::testing:Test {
protected:
	void SetUp() override {
		2dtree.add(2dcoords1, b1);
		2dtree.add(2dcoords2, b2);
		2dtree.add(2dcoords3, b3);

		3dtree.add(3dcoords1, v1);
		3dtree.add(3dcoords2, v2);
		3dtree.add(3dcoords3, v3);
	}

	KDTree<2, std::string> 2dtree;
	std::array<double, 2> 2dcoords1 = {0, 0};
	std::array<double, 2> 2dcoords2 = {-1, 1};
	std::array<double, 2> 2dcoords3 = {1, 0};
	std::string b1 = "building1";
	std::string b2 = "building2";
	std::string b3 = "building3";

	KDTree<3, std::string> 3dtree;
	std::array<double, 3> 3dcoords1 = {0, 0, 0};
	std::array<double, 3> 3dcoords2 = {-1, 0, 0};
	std::array<double, 3> 3dcoords3 = {-1, 1, 0};
	std::string v1 = "lyrebird";
	std::string v2 = "kolibri";
	std::string v3 = "eagle";
};

TEST_F(KDTreeTest, AddWorks) {
	EXPECT_EQ(2dtree.size(), 3);
	EXPECT_EQ(2dtree.height(), 1);

	EXPECT_EQ(3dtree.size(), 3);
	EXPECT_EQ(3dtree.height(), 2);
}

TEST_F(KDTreeTest, ContainsWorks) {
	std::string building = "building2";
	std::string bird = "eagle";

	EXPECT_EQ(2dtree.contains(building), true);
	EXPECT_EQ(2dtree.contains(bird), false);

	EXPECT_EQ(3dtree.contains(bird), true);
	EXPECT_EQ(3dtree.contains(building), false);

}

