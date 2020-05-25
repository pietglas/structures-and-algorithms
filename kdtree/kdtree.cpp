#include <array>
#include <cmath>


template<size_t N, typename T>
class KDTree {
private:
	struct Node {
		Node(std::array<double> coordinates, T data) : coordinates{coordinates},
			data{data} {}
		std::array<double> coordinates;
		T data;
		Node * left;
		Node * right;
	};
public:
	KDTree();
	~KDTree();

	KDTree(const KDTree & rhs);
	KDTree & operator =(const KDTree & rhs);

	Node * copy(Node * node) const;		// copies subtree with node as root
	Node * copyTree() const;	// copies the whole tree
	bool add(std::array<double> coords, T data);
	std::array<double> containsData(const T & data) const;
	bool containsCoordinates(const std::array<double> & coords) const;

private:
	Node * root_;
	void erase(Node * node);	// erases node and all its children
	bool addContainsHelper(std::array<double> coords, T data, 
		Node * node, size_t height, bool add=true);
	bool containsDataHelper(const T & data, Node * node);
};

template<size_t N, typename T>
KDTree<N, T>::KDTree() : root_{nullptr} {}

template<size_t N, typename T>
KDTree<N, T>::~KDTree() {
	KDTree<N, T>::erase(root_);
}

template<size_t N, typename T>
KDTree<N, T>::KDTree(const KDTree & rhs) {
	KDTree<N, T>::erase(root_);
	root_ = rhs.copyTree();
}

template<size_t N, typename T>
Node * KDTree<N, T>::copy(Node * node) {
	if (node != nullptr) {
		Node * new_node = new Node(node->coordinates, node->data);
		new_node->left = KDTree<N, T>::copy(node->left);
		new_node->right = KDTree<N, T>::copy(node->right);
		return new_node;
	}
	return nullptr;
}

template<size_t N, typename T>
Node * KDTree<N, T>::copyTree() {
	return KDTree<N, T>::copy(root_);
}

template<size_t N, typename T>
bool KDTree<N, T>::add(std::array<double> coords, T data) {
	return KDTree<N, T>::addHelper(coords, data, root_, 0);
}

template<size_t N, typename T>
std::array<double> KDTree<N, T>::containsData(const T & data) const {
	return KDTree<N, T>::containsDataHelper(data, root_);
}	

template<size_t N, typename T>
bool KDTree<N, T>::containsCoordinates(const std::array<double> & coords) const {
	return KDTree<N, T>::addHelper(coords, data, root_, 0, false);
}

template<size_t N, typename T>
void KDTree<N, T>::erase(Node * node) {
	if (node != nullptr) {
		erase(node->left);
		erase(node->right);
		delete node;
	}
}

template<size_t N, typename T>
bool KDTree<N, T>::addContainsHelper(std::array<double> coords, T data, 
		Node * node, size_t height, bool add) {
	if (node == nullptr) {
		node = new Node(coords, data);
		return false;
	}
	else {
		size_t index = height % N;
		bool contains;
		if (coords == node->coordinates)
			contains = true;
		else if (coords[index] < node->coordinates[index])
			contains = KDTree<N, T>::addHelper(coords, data, 
						node->left, height + 1, add);
		else
			contains = KDTree<N, T>::addHelper(coords, data, 
						node->right, height + 1, add);
		return contains;
	}

}

template<size_t N, typename T>
bool KDTree<N, T>::containsDataHelper(const T & data, Node * node) {
	if (node != nullptr) {
		if (node->data != data) 
			return std::min(KDTree<N, T>::containsDataHelper(data, node->left), 
							KDTree<N, T>::containsDataHelper(data, node->right));
		return true;
	}
	return false
}


