#include <unordered_map>
#include <stdexcept>

template <typename T, bool minpq>
class BoundedExtrinsicPQ {
	struct PriorityNode {
		PriorityNode() {};
		PriorityNode(T data, double priority) : data{data}, priority{priority} {}
		T data;
		double priority;
	}

public:
	BoundedExtrinsicPQ(size_t bound);
	BoundedExtrinsicPQ(const BoundedExtrinsicPQ & rhs);
	~BoundedExtrinsicPQ();

	BoundedExtrinsicPQ& operator =(const BoundedExtrinsicPQ& rhs);

	// accessors
	T& top() const;
	void empty() const;
	size_t size() const;

	// modifiers
	void push(T item, double priority);
	T pop();
	void changePriority(T item, double priority);

private:
	PriorityNode * heap_;
	std::unordered_map<T, double> elts_;
	size_t size_;
	size_t silent_size_;
	size_t bound_;

	void pushUp(int pos);
	void pushDown(int pos);
	bool contains(T item);
	void swap(int pos_1, int pos_2);
	bool compare(double priority_1, double priority_2);
	void resize(bool up=true);
};

template<typename T, bool minpq>
BoundedExtrinsicPQ::BoundedExtrinsicPQ(size_t bound) {
	bound_ = bound;
	size_ = 0;
	silent_size_ = 8;
	heap_ = new PriorityNode[silent_size_];
}

BoundedExtrinsicPQ::BoundedExtrinsicPQ(const BoundedExtrinsicPQ<T, minpq>& rhs) {
	if (this != &rhs) {
		size_ = rhs.size_;
		silent_size_ = rhs.silent_size_;
		bound_ = rhs.bound_;
		heap_ = new PriorityNode[silent_size_];
		for (int i = 0; i != size_; i++)
			heap_[i] = rhs.heap_[i];
	}
}

BoundedExtrinsicPQ::~BoundedExtrinsicPQ() {delete[] heap_;}

BoundedExtrinsicPQ<T, minpq>& 
	BoundedExtrinsicPQ::operator =(const BoundedExtrinsicPQ<T, minpq>& rhs) {
	if (this != &rhs) {
		delete[] heap_;
		size_ = rhs.size_;
		silent_size_ = rhs.silent_size_;
		bound_ = rhs.bound_;
		heap_ = new PriorityNode[silent_size_];
		for (int i = 0; i != size_; i++)
			heap_[i] = rhs.heap_[i];
	}
	return *this;
}

T& BoundedExtrinsicPQ::top() const {return heap_[0];}

void BoundedExtrinsicPQ::empty() const{return size_ == 0;}

size_t BoundedExtrinsicPQ::size() const {return size_;}

void BoundedExtrinsicPQ::push(T item, double priority) {
	heap_[size_] = PriorityNode(item, priority);

}

T BoundedExtrinsicPQ::pop() {
	T data = top();
	size_--;
	swap(0, size_);
	pushDown(0);
	if (size_ < (silent_size_ / 4))
		resize(false);
	return data;
}
	

void BoundedExtrinsicPQ::changePriority(T item, double priority) {

}

void pushUp(int pos);
void pushDown(int pos);
bool contains(T item);
void swap(int pos_1, int pos_2);
bool compare(double priority_1, double priority_2);
void resize(bool up=true);