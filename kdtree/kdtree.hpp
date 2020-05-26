#pragma once

#include <array>
#include <cmath>


template<size_t N, typename T>
class KDTree {
private:
	struct Node {
		Node(std::array<double, N> coordinates, T data) : coordinates{coordinates},
			data{data} {}
		std::array<double, N> coordinates;
		T data;
		Node * left;
		Node * right;
	};
public:
	KDTree();
	~KDTree();

	KDTree(const KDTree & rhs);
	KDTree & operator =(const KDTree & rhs);

	size_t size() const;
	size_t height() const;
	Node * copy(Node * node) const;		// copies subtree with node as root
	Node * copyTree() const;	// copies the whole tree
	bool add(std::array<double, N> coords, T data);
	std::array<double, N> containsData(const T & data) const;
	bool containsCoordinates(const std::array<double, N> & coords) const;

private:
	Node * root_;
	size_t size_;
	size_t height_;
	void erase(Node * node);	// erases node and all its children
	bool addContainsHelper(std::array<double, N> coords, T data, 
		Node * node, size_t height, bool add=true);
	bool containsDataHelper(const T & data, Node * node);
};

template<size_t N, typename T>
KDTree<N, T>::KDTree() : root_{nullptr}, size_{0}, height_{0} {}

template<size_t N, typename T>
KDTree<N, T>::~KDTree() {
	KDTree<N, T>::erase(root_);
}

template<size_t N, typename T>
KDTree<N, T>::KDTree(const KDTree & rhs) {
	KDTree<N, T>::erase(root_);
	root_ = rhs.copyTree();
	size_ = rhs.size();
	height_ = rhs.height();
}

template<size_t N, typename T>
KDTree<N, T> & KDTree<N, T>::operator = (const KDTree & rhs) {
	if (this != rhs) {
		KDTree<N, T>::erase(root_);
		root_ = rhs.copyTree();
		size_ = rhs.size();
		height_ = rhs.height();
	}
	return this;
}

template<size_t N, typename T>
size_t KDTree<N, T>::size() const {
	return size_;
}

template<size_t N, typename T>
size_t KDTree<N, T>::height() const {
	return height_;
}

template<size_t N, typename T>
KDTree<N, T>::Node * KDTree<N, T>::copy(KDTree<N, T>::Node * node) {
	if (node != nullptr) {
		KDTree<N, T>::Node * new_node = 
			new KDTree<N, T>::Node(node->coordinates, node->data);
		new_node->left = KDTree<N, T>::copy(node->left);
		new_node->right = KDTree<N, T>::copy(node->right);
		return new_node;
	}
	return nullptr;
}

template<size_t N, typename T>
KDTree<N, T>::Node * KDTree<N, T>::copyTree() {
	return KDTree<N, T>::copy(root_);
}

template<size_t N, typename T>
bool KDTree<N, T>::add(std::array<double, N> coords, T data) {
	return KDTree<N, T>::addHelper(coords, data, root_, 0);
}

template<size_t N, typename T>
std::array<double, N> KDTree<N, T>::containsData(const T & data) const {
	return KDTree<N, T>::containsDataHelper(data, root_);
}	

template<size_t N, typename T>
bool KDTree<N, T>::containsCoordinates(const std::array<double, N> & coords) const {
	return KDTree<N, T>::addHelper(coords, data, root_, 0, false);
}

template<size_t N, typename T>
void KDTree<N, T>::erase(KDTree<N, T>::Node * node) {
	if (node != nullptr) {
		erase(node->left);
		erase(node->right);
		delete node;
	}
}

template<size_t N, typename T>
bool KDTree<N, T>::addContainsHelper(std::array<double, N> coords, T data, 
		KDTree<N, T>::Node * node, size_t height, bool add) {
	if (node == nullptr) {
		node = new Node(coords, data);
		++size_;
		if (height > height_)
			++height_;
		return false;
	}
	else {
		size_t index = height % N;
		if (coords == node->coordinates)
			return true;
		else if (coords[index] < node->coordinates[index])
			return KDTree<N, T>::addHelper(coords, data, 
						node->left, height + 1, add);
		else
			return KDTree<N, T>::addHelper(coords, data, 
						node->right, height + 1, add);
	}

}

template<size_t N, typename T>
bool KDTree<N, T>::containsDataHelper(const T & data, KDTree<N, T>::Node * node) {
	if (node != nullptr) {
		if (node->data != data) 
			return std::min(KDTree<N, T>::containsDataHelper(data, node->left), 
							KDTree<N, T>::containsDataHelper(data, node->right));
		return true;
	}
	return false
}


