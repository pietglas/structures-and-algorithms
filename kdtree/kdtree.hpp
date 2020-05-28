#pragma once

#include <array>
#include <cmath>
#include <iostream>


template<size_t N, typename T>
class KDTree {

	struct Node {
		Node(std::array<double, N> coordinates, T data) : coordinates{coordinates},
			data{data} {}
		std::array<double, N> coordinates;
		T data;
		Node * left = nullptr;
		Node * right = nullptr;
	};

public:
	KDTree();
	~KDTree();

	KDTree(const KDTree & rhs);
	KDTree & operator =(const KDTree & rhs);

	size_t size() const;
	bool empty() const; 
	int height() const;
	// add data at a point. In case where the three already stores
	// data at the point, it does nothing and returns false. 
	bool add(std::array<double, N> coords, T data);
	bool contains(const T & data) const;
	void print() const;

	// returns a reference to the data at the point
	T & data operator [](std::array<double, N> coords);

private:
	Node * root_;
	size_t size_;
	int height_;
	Node * copyTree() const;	// copies the whole tree
	void erase(Node * node);	// erases node and all its children
	Node * copy(Node * node) const;		// copies subtree with node as root
	bool addHelper(std::array<double, N> coords, T data, 
		Node *& node, int height);
	bool containsHelper(const T & data, Node * node) const;
	void printHelper(Node * node) const;
};

template<size_t N, typename T>
KDTree<N, T>::KDTree() : root_{nullptr}, size_{0}, height_{-1} {}

template<size_t N, typename T>
KDTree<N, T>::~KDTree() {
	erase(root_);
}

template<size_t N, typename T>
KDTree<N, T>::KDTree(const KDTree & rhs) {
	root_ = rhs.copyTree();
	size_ = rhs.size();
	height_ = rhs.height();
}

template<size_t N, typename T>
KDTree<N, T> & KDTree<N, T>::operator =(const KDTree & rhs) {
	if (this != &rhs) {
		erase(root_);
		root_ = rhs.copyTree();
		size_ = rhs.size();
		height_ = rhs.height();
	}
	return *this;
}

template<size_t N, typename T>
size_t KDTree<N, T>::size() const {
	return size_;
}

template<size_t N, typename T>
bool KDTree<N, T>::empty() const {
	return size_ == 0;
}

template<size_t N, typename T>
int KDTree<N, T>::height() const {
	return height_;
}

template<size_t N, typename T>
typename KDTree<N, T>::Node * KDTree<N, T>::copy(Node * node) const {
	if (node != nullptr) {
		Node * new_node = new Node(node->coordinates, node->data);
		new_node->left = copy(node->left);
		new_node->right = copy(node->right);
		return new_node;
	}
	return nullptr;
}

template<size_t N, typename T>
typename KDTree<N, T>::Node * KDTree<N, T>::copyTree() const {
	return copy(root_);
}

template<size_t N, typename T>
bool KDTree<N, T>::add(std::array<double, N> coords, T data) {
	return addHelper(coords, data, root_, 0);
}

template<size_t N, typename T>
bool KDTree<N, T>::contains(const T & data) const {
	return containsHelper(data, root_);
}	

template<size_t N, typename T>
void KDTree<N, T>::print() const {
	std::cout << "size tree: " << size_ << std::endl;
	std::cout << "height tree: " << height_ << std::endl;
	printHelper(root_);
}

template<size_t N, typename T>
T & data KDTree<N, T>::operator [](std::array<double, N> coords) {

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
bool KDTree<N, T>::addHelper(std::array<double, N> coords, T data, 
		Node *& node, int height) {
	if (node == nullptr) {
		node = new Node(coords, data);
		size_++;
		if (height > height_)
			height_ = height;
		return true;
	}
	else {
		size_t index = height % N;
		if (coords == node->coordinates)
			return false;
		else {
			if (coords[index] < node->coordinates[index])
				return addHelper(coords, data, 
							node->left, height + 1);
			return addHelper(coords, data, 
						node->right, height + 1);
		}
	}
}

template<size_t N, typename T>
bool KDTree<N, T>::containsHelper(const T & data, Node * node) const {
	if (node != nullptr) {
		if (node->data != data) 
			return std::max(containsHelper(data, node->left), 
							containsHelper(data, node->right));
		return true;
	}
	return false;
}

template<size_t N, typename T>
void KDTree<N, T>::printHelper(Node * node) const {
	if (node != nullptr) {
		std::cout << "data: " << node->data << std::endl;
		printHelper(node->left);
		printHelper(node->right);
	}
}

template<size_t N, typename T>
T & KDTree<N, T>::bracketHelper(std::array<double, N> coords, Node * node,
		bool found) {
	if (node->coordinates == coords)
		return node->data;
	else {
		T data = 
	}
}


