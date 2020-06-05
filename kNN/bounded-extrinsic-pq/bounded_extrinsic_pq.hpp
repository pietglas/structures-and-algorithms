#pragma once

#include <stdexcept>

template <typename T, bool minpq>
class BoundedExtrinsicPQ {
	struct PriorityNode {
		PriorityNode() {};
		PriorityNode(T data, double priority) : data{data}, priority{priority} {}
		T data;
		double priority;
	};

public:
	BoundedExtrinsicPQ(int bound);
	BoundedExtrinsicPQ(const BoundedExtrinsicPQ & rhs);
	~BoundedExtrinsicPQ();

	BoundedExtrinsicPQ& operator =(const BoundedExtrinsicPQ& rhs);

	// accessors
	T& top() const;
	bool empty() const;
	int size() const;

	// modifiers
	void push(T item, double priority);
	T pop();

private:
	PriorityNode * heap_ = nullptr;
	int size_ = 0;
	int silent_size_ = 8;
	int bound_;

	void pushUp(int pos);
	void pushDown(int pos);
	void swap(int pos_1, int pos_2);
	bool compare(double priority_1, double priority_2);
	void resize(bool up=true);
};

template<typename T, bool minpq>
BoundedExtrinsicPQ<T, minpq>::BoundedExtrinsicPQ(int bound) {
	bound_ = bound;
	heap_ = new PriorityNode[silent_size_];
}

template <typename T, bool minpq>
BoundedExtrinsicPQ<T, minpq>::BoundedExtrinsicPQ(const BoundedExtrinsicPQ<T, minpq>& rhs) {
	size_ = rhs.size_;
	silent_size_ = rhs.silent_size_;
	bound_ = rhs.bound_;
	heap_ = new PriorityNode[silent_size_];
	for (int i = 0; i != size_; i++)
		heap_[i] = rhs.heap_[i];
}

template <typename T, bool minpq>
BoundedExtrinsicPQ<T, minpq>::~BoundedExtrinsicPQ() {delete[] heap_;}

template <typename T, bool minpq>
BoundedExtrinsicPQ<T, minpq>& 
	BoundedExtrinsicPQ<T, minpq>::operator =(const BoundedExtrinsicPQ<T, minpq>& rhs) {
	if (this != &rhs) {
		if (heap_ != nullptr)
			delete[] heap_;
		size_ = rhs.size_;
		silent_size_ = rhs.silent_size_;
		bound_ = rhs.bound_;
		heap_ = new PriorityNode[silent_size_];
		for (int i = 0; i != size_; i++)
			heap_[i] = PriorityNode(rhs.heap_[i].data, rhs.heap_[i].priority);
	}
	return *this;
}

template <typename T, bool minpq>
T& BoundedExtrinsicPQ<T, minpq>::top() const {return heap_[0].data;}

template <typename T, bool minpq>
bool BoundedExtrinsicPQ<T, minpq>::empty() const{return size_ == 0;}

template <typename T, bool minpq>
int BoundedExtrinsicPQ<T, minpq>::size() const {return size_;}

template <typename T, bool minpq>
void BoundedExtrinsicPQ<T, minpq>::push(T item, double priority) {
	if (size_ == silent_size_)
		resize();
	heap_[size_] = PriorityNode(item, priority);
	pushUp(size_);
	if (size_ == bound_)
		pop();
	++size_;
}

template <typename T, bool minpq>
T BoundedExtrinsicPQ<T, minpq>::pop() {
	if (empty())
		throw std::out_of_range("pq is empty already");
	T data = top();
	size_--;
	swap(0, size_);
	pushDown(0);
	if (size_ < (silent_size_ / 4))
		resize(false);
	return data;
}

template <typename T, bool minpq>
void BoundedExtrinsicPQ<T, minpq>::pushUp(int pos) {
	int parent_pos;
	if (pos % 2 == 0)
		parent_pos = pos / 2 - 1;
	else
		parent_pos = pos / 2;
	if (parent_pos >= 0 && compare(heap_[pos].priority, heap_[parent_pos].priority)) {
		swap(pos, parent_pos);
		pushUp(parent_pos);
	}
}

template <typename T, bool minpq>
void BoundedExtrinsicPQ<T, minpq>::pushDown(int pos) {
	int child1 = 2 * pos + 1;
	int child2 = 2 * pos + 2;
	int minmaxchild;
	if (child2 < size_) {
		if (compare(heap_[child1].priority, heap_[child2].priority))
			minmaxchild = child1;
		else
			minmaxchild = child2;
	} 
	else if (child2 == size_)
		minmaxchild = child1;
	else
		return;
	if (compare(heap_[minmaxchild].priority, heap_[pos].priority)) {
		swap(minmaxchild, pos);
		pushDown(minmaxchild);
	}
}

template <typename T, bool minpq>
void BoundedExtrinsicPQ<T, minpq>::swap(int pos_1, int pos_2) {
	PriorityNode temp = heap_[pos_1];
	heap_[pos_1] = heap_[pos_2];
	heap_[pos_2] = temp;
}

template <typename T, bool minpq>
bool BoundedExtrinsicPQ<T, minpq>::compare(double priority_1, double priority_2) {
	if (minpq)
		return priority_1 < priority_2;
	else
		return priority_1 > priority_2;
}

template <typename T, bool minpq>
void BoundedExtrinsicPQ<T, minpq>::resize(bool up) {
	if (up)
		silent_size_ *= 2;
	else
		silent_size_ /= 2;
	PriorityNode * new_heap = new PriorityNode[silent_size_];
	for (int i = 0; i != size_; i++)
		new_heap[i] = heap_[i];
	delete[] heap_;
	heap_ = new_heap;
	new_heap = nullptr;
}