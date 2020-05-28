#include "../kdtree.hpp"
#include <string>
#include <array>
#include <gtest/gtest.h>

class KDTreeTest : public ::testing::Test {
protected:
	void SetUp() override {
		twodtree.add(twodcoords1, b1);
		twodtree.add(twodcoords2, b2);
		twodtree.add(twodcoords3, b3);

		threedtree.add(threedcoords1, v1);
		threedtree.add(threedcoords2, v2);
		threedtree.add(threedcoords3, v3);

		twodtree2.add(twodcoords1, b2);
		twodtree2 = twodtree;
	}

	KDTree<2, std::string> twodtree;
	std::array<double, 2> twodcoords1 = {0, 0};
	std::array<double, 2> twodcoords2 = {-1, 1};
	std::array<double, 2> twodcoords3 = {1, 0};
	std::string b1 = "building1";
	std::string b2 = "building2";
	std::string b3 = "building3";

	KDTree<3, std::string> threedtree;
	std::array<double, 3> threedcoords1 = {0, 0, 0};
	std::array<double, 3> threedcoords2 = {-1, 0, 0};
	std::array<double, 3> threedcoords3 = {-1, 1, 0};
	std::string v1 = "lyrebird";
	std::string v2 = "kolibri";
	std::string v3 = "eagle";

	KDTree<2, std::string> twodtree2;
};

TEST_F(KDTreeTest, AddWorks) {
	EXPECT_EQ(twodtree.size(), 3);
	EXPECT_EQ(twodtree.height(), 1);

	EXPECT_EQ(threedtree.size(), 3);
	EXPECT_EQ(threedtree.height(), 2);
}

TEST_F(KDTreeTest, ContainsWorks) {
	EXPECT_EQ(twodtree.contains(b1), true);
	EXPECT_EQ(twodtree.contains(b2), true);
	EXPECT_EQ(twodtree.contains(b3), true);

	EXPECT_EQ(threedtree.contains(v1), true);
	EXPECT_EQ(threedtree.contains(v2), true);
	EXPECT_EQ(threedtree.contains(v3), true);
}

TEST_F(KDTreeTest, CopyWorks) {
	EXPECT_EQ(twodtree2.size(), 3);
	EXPECT_EQ(twodtree2.height(), 1);
	EXPECT_EQ(twodtree2.contains(b1), true);
	EXPECT_EQ(twodtree2.contains(b2), true);
	EXPECT_EQ(twodtree2.contains(b3), true);
}

