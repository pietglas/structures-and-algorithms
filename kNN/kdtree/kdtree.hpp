#pragma once

#include <array>
#include <cmath>	// std::max, std::sqrt, 
#include <iostream>
#include <vector>
#include <map>
#include "../bounded-extrinsic-pq/bounded_extrinsic_pq.hpp"

/* Implementation of a kdtree. The dimension as well as the data type
 * are to be specified as template arguments. It contains some standard
 * functionality such as adding an element and checking if an element is 
 * in the tree. 
 * It also contains an implementation of a kNN search (see kNN), 
 * an algorithm to find the k nearest neighbors of a given point. 
 */
template<size_t N, typename T>
class KDTree {
private:
	struct Node {
		Node(std::array<double, N> coordinates) : coordinates{coordinates} {}
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

	KDTree(const KDTree& rhs);
	KDTree & operator =(const KDTree& rhs);

	KDTree(const std::map<std::array<double, N>, T>& data);

	size_t size() const;
	bool empty() const; 
	int height() const;
	/** add data at a point. In case where the three already stores
	    data at the point, it does nothing and returns false. */
	bool add(const std::array<double, N>& coords, const T& data);
	bool contains(const T& data) const;
	void print() const;

	/** access or insert data */
	T& operator [](const std::array<double, N>& coords);

	/** k-nearest neighbor algorithm. Returns a vector with the specified
	    number of neighbors (or less, if the specified number is larger than
	    the number of data points). */
	std::vector<T> kNN(const std::array<double, N>& coords, T item,
			int nr_neighbors);

private:
	Node * root_;
	size_t size_;
	int height_;

	Node * copy(Node * node) const;		// copies subtree with node as root
	Node * copyTree() const;	// copies the whole tree
	void erase(Node * node);	// erases node and all its children
	
	bool addHelper(const std::array<double, N>& coords, const T& data, 
		Node *& node, int height);
	bool containsHelper(const T& data, Node * node) const;
	void printHelper(Node * node) const;
	T& bracketHelper(const std::array<double, N>& coords, Node *& node, int height);

	double distance(const std::array<double, N>& coords0, 
		const std::array<double, N>& coords1);

	void kNNHelper(const std::array<double, N>& coords,
		BoundedExtrinsicPQ<T, false>& neighbors, Node * node, int height);
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
KDTree<N, T>::KDTree(const std::map<std::array<double, N>, T>& data) {
	root_ = nullptr;
	size_ = 0;
	height_ = -1;
	for (auto const& item : data)
		add(item.first, item.second);
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
bool KDTree<N, T>::add(const std::array<double, N>& coords, const T& data) {
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
T& KDTree<N, T>::operator [](const std::array<double, N>& coords) {
	return bracketHelper(coords, root_, 0);
}

template<size_t N, typename T>
std::vector<T> KDTree<N, T>::kNN(const std::array<double, N>& coords, T item,
			int nr_neighbors) {
	// max priority queue containing the temporary neighbors
	BoundedExtrinsicPQ<T, false> neighbors{nr_neighbors};
	// call helper function that does the actual work
	kNNHelper(coords, neighbors, root_, 0);
	// return a sorted vector with the neighbors, starting with the 
	// closest neighbor (which is the last element to be removed from
	// our max pq)
	std::vector<T> neighbors_sorted;
	neighbors_sorted.reserve(neighbors.size());
	nr_neighbors = neighbors.size();
	for (int i = 0; i != nr_neighbors; i++)
		neighbors_sorted.push_back(neighbors.pop());
	return neighbors_sorted;
}

template<size_t N, typename T>
void KDTree<N, T>::kNNHelper(const std::array<double, N>& coords,
		BoundedExtrinsicPQ<T, false>& neighbors, Node * node, int height) {
	if (node != nullptr) {
		int index = height % N;
		double distance_node = distance(node->coordinates, coords);
		// add data current node to the priority queue, if is closer
		// than the current most far away selected neighbor, or if
		// the max capacity of the pq hasn't been reached
		if (distance_node < neighbors.topPriority() || 
				neighbors.size() < neighbors.bound())
			neighbors.push(node->data, distance_node);
		// determine which of the children is more promosing wrt distance
		Node * promosing_node = node->right;
		Node * other_node = node->left;
		if (coords[index] < node->coordinates[index]) {	// swap if needed
			promosing_node = node->left;
			other_node = node->right;
		}
		kNNHelper(coords, neighbors, promosing_node, height + 1);
		// check for possible closer values in the other direction
		std::array<double, N> closest_point_rs = coords;
		closest_point_rs[index] = node->coordinates[index];
		// the following calculation is actually redundant in the current
		// approach, as only one index differs
		double closest_distance_rs = distance(closest_point_rs, coords);
		if (closest_distance_rs < neighbors.topPriority() || 
				neighbors.size() < neighbors.bound())
			kNNHelper(coords, neighbors, other_node, height + 1);
	}
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
bool KDTree<N, T>::addHelper(const std::array<double, N>& coords, const T& data, 
		Node *& node, int height) {
	if (node == nullptr) {
		node = new Node(coords, data);
		size_++;
		if (height > height_)
			height_ = height;
		return true;
	}
	size_t index = height % N;
	if (coords == node->coordinates)
		return false;
	if (coords[index] < node->coordinates[index])
		return addHelper(coords, data, 
					node->left, height + 1);
	return addHelper(coords, data, 
				node->right, height + 1);
}

template<size_t N, typename T>
bool KDTree<N, T>::containsHelper(const T& data, Node * node) const {
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
T& KDTree<N, T>::bracketHelper(const std::array<double, N>& coords, Node *& node,
		int height) {
	if (node != nullptr) {
		if (node->coordinates == coords)
			return node->data;
		size_t index = height % N;
		if (coords[index] < node->coordinates[index])
			return bracketHelper(coords, node->left, height + 1);
		return bracketHelper(coords, node->right, height + 1);
	}
	node = new Node(coords);
	return node->data;
}

template<size_t N, typename T>
double KDTree<N, T>::distance(const std::array<double, N>& coords0, 
		const std::array<double, N>& coords1) {
	double sum = 0;
	for (int i = 0; i != N; i++) {
		double summand = pow(coords0[i] - coords1[i], 2);
		sum += summand;
	}
	return std::sqrt(sum);
}


