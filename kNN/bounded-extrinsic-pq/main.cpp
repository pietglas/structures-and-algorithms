#include "bounded_extrinsic_pq.hpp"
#include <string>

int main() {
	BoundedExtrinsicPQ<std::string, true> pq0{4};
	BoundedExtrinsicPQ<std::string, true> pq1{2};
	BoundedExtrinsicPQ<std::string, true> pq2{2};
	BoundedExtrinsicPQ<int, true> pq3{12};
	std::string str0 = "plato";
	std::string str1 = "aristotle";
	std::string str2 = "kant";
	std::string str3 = "hegel";

	pq0.push(str0, 1);
	pq0.push(str1, 2);
	pq0.push(str2, 0);
	pq0.push(str3, 3);

	pq1.push(str0, 2);
	pq1.push(str1, 1);
	pq1.push(str2, 1.5);

	pq2 = pq1;
	pq2.push(str2, 0);

	for (int i = 0; i != 15; i++)
		pq3.push(i, i);
	for (int i = 0; i != 11; i ++)
		pq3.pop();
}