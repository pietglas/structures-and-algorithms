#include "../bounded_extrinsic_pq.hpp"
#include <string>
#include <gtest/gtest.h>

class BoundedExtrinsicPQTest : public ::testing::Test {
protected:
	void SetUp() override {
		pq0.push(str0, 1);
		pq0.push(str1, 2);
		pq0.push(str2, 0);
		pq0.push(str3, 3);

		pq1.push(str0, 2);
		pq1.push(str1, 1);
		pq1.push(str2, 1.5);

		pq2 = pq1;

		for (int i = 0; i != 15; i++)
			pq3.push(i, i);
		for (int i = 0; i != 11; i ++)
			pq3.pop();
	}

	BoundedExtrinsicPQ<std::string, true> pq0{4};
	BoundedExtrinsicPQ<std::string, false> pq1{3};
	BoundedExtrinsicPQ<std::string, false> pq2{3};
	BoundedExtrinsicPQ<int, true> pq3{12};
	std::string str0 = "plato";
	std::string str1 = "aristotle";
	std::string str2 = "kant";
	std::string str3 = "hegel";
};

TEST_F(BoundedExtrinsicPQTest, AddWorks) {
	EXPECT_EQ(pq1.size(), 3);
	EXPECT_EQ(pq0.size(), 4);
	EXPECT_EQ(pq0.top(), str2);
}

TEST_F(BoundedExtrinsicPQTest, CopyWorks) {
	EXPECT_EQ(pq2.size(), pq1.size());
	EXPECT_EQ(pq1.top(), pq2.top());
}

TEST_F(BoundedExtrinsicPQTest, PopWorks) {
	pq2.push(str2, 0);
	EXPECT_EQ(pq2.top(), str2);
}

TEST_F(BoundedExtrinsicPQTest, ResizeWorks) {
	EXPECT_EQ(pq3.size(), 1);
	EXPECT_EQ(pq3.top(), 11);
}