#include <array>

template<size_t N, typename T>
class KDTree {
private:
	struct Node {
		Node(std::array<double> coordinates, T data) : coordinates_{coordinates},
			data_{data} {}
		std::array<double> coordinates_;
		T data_;
		Node * left_;
		Node * right_;
	};
public:
	KDTree();
	~KDTree();

	KDTree(const KDTree & rhs);
	KDTree & operator =(const KDTree & rhs);

	void add(std::array<double> coordinates, T data);
	std::array<double> containsData(T data);
	bool containsCoordinates(std::array<double> coordinates);

private:
	Node * root_;
};

template<size_t N, typename T>
KDTree<N, T>::KDTree() : root_{nullptr} {}

template<size_t N, typename T>
KDTree<N, T>::~KDTree() {

}


