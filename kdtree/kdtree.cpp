#include <array>

template<size_t N, typename Type>
class KDTree {
public:
	struct Node {
		Node()
		Node(std::array coordinates, )
		std::array coordinates_;
	};
	KDTree();
	~KDTree();

	KDTree(const KDTree & rhs);
	KDTree & operator =(const KDTree & rhs);

	void add()
};