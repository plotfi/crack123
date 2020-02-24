#include <algorithm>
#include <bitset>
#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <stack>
#include <unordered_map>
#include <vector>

// ** Behavioral Prep Grid:
//
// Question          Project1    Project2    Project3
// Challenges
// Mistake/Fails
// Enjoyed
// Leadership
// Conflicts
// What to do Diff

// ** What are your weaknesses?

// ** Big O
// * Recursive functions are often O(branches^depth)
// * fully balanced tree is O(2^logN)

// Use proof by induction..
// *
// **
// ***
// ****
//
// *****  ****
// ******  ***
// *******  **
// ********  *
// *********
//
// (10 * 9) /2
//
// O((N * (N-1))/2)
//
// TODO: Resume Big O at Page 52

// >>> Page 60: Technical Questions
//
// Core Data Structures: Linked List, Trees, Tries, Graphs, Stacks, Queues,
//                       Heaps, Vectors, Hash Tables.
// TODO: Study up LLVM ADTs
// Algos: BFS, DFS, Binary Search, Merge Sort, Quick Sort, Bucket Sort
// Cocepts: Bit Manip, Memory (stack vs heap), Recursion, Dynamic Programming,
//          Big O Time and Space

//// ** Question 1.1, Algo to determine if a string has unique chars,
//                    what if you cant use datastructures?
// Bit table technique
std::string hasUniqueChars(const std::string &str) {
  std::bitset<256> bits;
  std::bitset<256> repeat;
  for (auto c : str)
    if (!bits.test(c) || repeat.test(c))
      bits[c] = 1;
    else
      repeat[c] = 1;

  std::string repeatChars;
  for (unsigned i = 0; i < repeat.size(); i++)
    if (repeat.test(i))
      repeatChars += (char)i;
  return repeatChars;
}

// sorting technique
std::string hasUniqueChars_2(const std::string &str) {
  std::string sortedStr = str;
  std::sort(sortedStr.begin(), sortedStr.end());

  std::string repeatChars;
  char prev = '\0';
  for (auto c : sortedStr) {
    if (repeatChars.size() == 0 || (c == prev && repeatChars.back() != c))
      repeatChars += c;
    prev = c;
  }
  return repeatChars;
}

// isUnique and bail
bool isUnique(const std::string &str) {
  if (str.length() > 256)
    return false;
  std::bitset<256> charSet;
  for (auto c : str)
    if (charSet.test(c))
      return false;
    else
      charSet.set(c);
  return true;
}

//// ** Question 1.2
bool isPermutation(const std::string &a, const std::string &b) {
  std::vector<size_t> charCountA;
  charCountA.resize(256, 0);
  for (auto c : a)
    charCountA[c]++;

  std::vector<size_t> charCountB;
  charCountB.resize(256, 0);
  for (auto c : b)
    charCountB[c]++;

  return std::equal(charCountA.begin(), charCountA.end(), charCountB.data());
}

bool isPermutation_2(const std::string &a, const std::string &b) {
  std::vector<int> count;
  count.resize(256, 0);
  for (auto c : a)
    count[c]++;

  for (auto c : b)
    count[c]--;

  return 0 == std::accumulate(count.begin(), count.end(), 0);
}

// 1.3
std::string URLify(const std::string &str) {
  const size_t slen = str.size() + 1;
  const size_t nlen = slen + (std::count(str.begin(), str.end(), ' ') * 3);
  char url[nlen];
  bzero(url, nlen);
  str.copy(url, slen - 1);
  std::copy_backward(url, url + slen, url + nlen);
  for (unsigned i = 0, j = 0; i <= slen; i++) {
    std::string s = [](const char c) {
      return c == ' ' ? "%20" : std::string(&c, 1);
    }(url[nlen - slen + i]);
    j += s.copy(&url[j], s.size());
  }
  return std::string(url);
}

// 1.4
bool isPalidromePermutation(const std::string &str) {
  std::bitset<256> bits;
  for (auto c : str)
    bits.flip(c);
  return bits.count() <= 1;
}

// 1.5
size_t editDistance(const std::string &ls, const std::string &ss) {
  return (ls.size() < ss.size()) ? editDistance(ss, ls) : [&ls, &ss]() {
    unsigned edits = 0;
    for (unsigned si = 0, li = 0; li < ls.size(); li++) {
      if (si > ss.size() || ls[li] != ss[si])
        edits++;
      if (ls[li] == ss[si] || ls.size() == ss.size())
        si++;
    }
    return edits;
  }();
}

// 1.6
std::string compress(std::string &str) {
  std::stringstream sstr;

  auto encode = [&sstr](unsigned count, char p) {
    sstr << count << (('0' <= p && p <= '9') ? "#" : "")
         << (p == '#' ? "#" : "") << p;
  };

  unsigned count = 0;
  char p = str[0];
  for (auto c : str) {
    if (c == p) {
      count++;
      continue;
    }

    encode(count, p);
    p = c;
    count = 1;
  }

  encode(count, p);
  return sstr.str();
}

std::string decompress(const std::string &str) {
  std::stringstream sstr;
  for (unsigned i = 0; i < str.length(); i++) {
    unsigned count = 0;
    for (; '0' <= str[i] && str[i] <= '9'; i++)
      count += (count * 10) + (str[i] - '0');
    i += (str[i] == '#');
    for (unsigned j = 0; j < count; j++)
      sstr << str[i];
  }
  return sstr.str();
}

// 1.7
// 1 2 3    7 4 1
// 4 5 6 -> 8 5 2
// 7 8 9    9 6 3

// 1 2 3    9 6 3    7 4 1
// 4 5 6 -> 8 5 2 -> 8 5 2
// 7 8 9    7 4 1    9 6 3

template <unsigned N> void printMat(int matrix[N][N]) {
  for (unsigned i = 0; i < N; i++) {
    for (unsigned j = 0; j < N; j++)
      std::cout << matrix[i][j] << " ";
    std::cout << "\n";
  }
  std::cout << "\n";
}

// 1.7
template <unsigned N> void rotateMat(int matrix[N][N]) {
  // flip along diagonal
  for (unsigned i = 0; i < N; i++)
    for (unsigned j = 0; j < N - i; j++)
      std::swap(matrix[i][j], matrix[N - 1 - j][N - 1 - i]);
  // flip along horizontal
  for (unsigned i = 0; i < N / 2; i++)
    for (unsigned j = 0; j < N; j++)
      std::swap(matrix[i][j], matrix[N - 1 - i][j]);
}

// 1.8
template <unsigned M, unsigned N> void zeroRowCols(int matrix[M][N]) {
  std::set<unsigned> rows;
  std::set<unsigned> cols;

  for (unsigned i = 0; i < M; i++)
    for (unsigned j = 0; j < N; j++)
      if (!matrix[i][j]) {
        rows.insert(i);
        cols.insert(j);
      }

  for (auto row : rows)
    for (unsigned j = 0; j < N; j++)
      matrix[row][j] = 0;
  for (auto col : cols)
    for (unsigned i = 0; i < M; i++)
      matrix[i][col] = 0;
}

// 1.9
bool isSubstring(const std::string &s1, const std::string &s2) { return false; }

struct LL {
  int data;
  LL *next;
  LL(int data) : data(data), next(nullptr) {}
  LL(int data, LL *next) : data(data), next(next) {}
};

template <typename T> struct LL2 {
  T data;
  std::unique_ptr<LL2<T>> next;
  LL2(T data) : data(data), next(nullptr) {}
  LL2(T data, std::unique_ptr<LL2<T>> next)
      : data(data), next(std::move(next)) {}
};

// 2.1
void dedup(LL *&head) {
  std::set<int> dedup;
  dedup.insert(head->data);
  if (!head)
    return;
  LL *prev = head;
  LL *curr = head->next;
  while (curr) {
    if (dedup.count(curr->data)) {
      prev->next = curr->next;
      curr = prev->next;
      continue;
    }
    dedup.insert(curr->data);
    prev = prev->next;
    curr = curr->next;
  }
}

// 2.2
int kthLast(LL *head, int k) {
  unsigned count = 0;
  for (LL *curr = head; curr; curr = curr->next)
    count++;
  for (LL *curr = head; curr; curr = curr->next)
    if ((count-- - k) <= 0)
      return curr->data;
  return head->data;
}

// 2.3
void removeNode(LL *&head, const LL *node) {
  if (!head || !node)
    return;

  for (LL *curr = head; curr; curr = curr->next)
    if (curr->next == node) {
      curr->next = curr->next->next;
      break;
    }

  if (head == node)
    head = head->next;

  delete (node);
}

// 2.4
// TODO Partition???

// 2.5
int sumListLittleEndian_toInt(LL *head1, LL *head2) {
  int sum = 0;
  int mul = 1;
  while (head1 || head2) {

    int val1 = 0;
    if (head1) {
      val1 = head1->data;
      head1 = head1->next;
    }

    int val2 = 0;
    if (head2) {
      val2 = head2->data;
      head2 = head2->next;
    }

    sum += (val1 + val2) * mul;
    mul *= 10;
  }
  return sum;
}

size_t LLLength(LL *head) {
  size_t len = 0;
  for (; head; head = head->next)
    len++;
  return len;
}

int sumListBigEndian_toInt(LL *head1, LL *head2) {

  size_t head1Len = LLLength(head1);
  size_t head2Len = LLLength(head2);
  // Assume always that head1Len >= head2Len.
  if (head1Len < head2Len)
    return sumListBigEndian_toInt(head2, head1);

  int sum = 0;
  while (head1Len != head2Len) {
    sum *= 10;
    sum += head1->data;
    head1Len--;
    head1 = head1->next;
  }

  while (head1 && head2) {
    sum *= 10;
    sum += head1->data + head2->data;
    head1 = head1->next;
    head2 = head2->next;
  }

  return sum;
}

LL *int_toSumListBigEndian(int sum) {
  LL *head = nullptr;
  for (; sum; sum /= 10)
    head = new LL(sum % 10, head);
  return head;
}

LL *int_toSumListLittleEndian(int sum) {
  LL *head = nullptr;
  LL *tail = nullptr;
  for (; sum; sum /= 10) {
    LL *curr = new LL(sum % 10);
    if (!tail) {
      head = tail = curr;
      continue;
    }

    tail->next = curr;
    tail = curr;
  }

  return head;
}

auto printLL = [](const LL *head) {
  for (const LL *curr = head; curr; curr = curr->next)
    std::cout << "[" << curr->data << "]->";
  std::cout << "//\n";
};

// 2.6
bool isLLPalindrome(LL *head) {
  if (!head)
    return true;

  auto revList = [](LL *head) {
    LL *stack = nullptr;
    while (head) {
      LL *curr = head;
      head = head->next;
      curr->next = stack;
      stack = curr;
    }
    return stack;
  };

  LL *mid = head;
  for (unsigned I = 0, MidI = LLLength(head) / 2; I <= MidI; I++)
    mid = mid->next;
  mid = revList(mid);

  bool isPal = true;
  for (LL *curr = mid; curr; curr = curr->next, head = head->next)
    isPal |= (curr->data == head->data);

  mid = revList(mid);
  return isPal;
}

// 2.7
LL *listIntersect(LL *head1, LL *head2) {
  if (!head1 || !head2)
    return nullptr;

  size_t len1 = LLLength(head1);
  size_t len2 = LLLength(head2);

  if (len1 < len2)
    return listIntersect(head2, head1);

  for (unsigned i = 0; i < len1 - len2; i++)
    head1 = head1->next;

  while (head1) {
    if (head1 == head2)
      return head1;
    head1 = head1->next;
    head2 = head2->next;
  }

  return nullptr;
}

// 2.7
LL *detectCycle(LL *head) {
  if (!head || !head->next)
    return nullptr;

  LL *slow = head;
  LL *fast = head->next;
  while (slow && fast && fast->next) {
    if (slow == fast)
      return slow;
    slow = slow->next;
    fast = fast->next->next;
  }

  return nullptr;
}

// 3.1
template <unsigned n> class MultiStack {
  std::vector<int> stack;
  std::vector<unsigned> tops;

  MultiStack() {
    stack.resize(n * 10);
    tops.resize(n);
  }

  void push(int a, unsigned i) {
    if (tops[i] >= stack.size() / n) {
      size_t oldSize = stack.size();
      stack.resize(oldSize * 2);
      for (unsigned i = n; i != ~0U; i--) {
        for (unsigned j = 0; j < oldSize / n; j++) {
          stack[(i * (stack.size() / n)) + j] = stack[(i * (oldSize / n)) + j];
        }
      }
    }

    stack[(i * (stack.size() / n)) + tops[i]] = a;
    tops[i]++;
  }

  int pop(unsigned i) {
    int a = stack[(i * (stack.size() / n)) + tops[i]];
    tops[i]--;
    return a;
  }

  int peak(unsigned i) { return stack[(i * (stack.size() / n)) + tops[i]]; }
};

struct BT {
  int data;
  BT *left;
  BT *right;
  BT(int data, BT *left, BT *right) : data(data), left(left), right(right) {}
  BT(int data) : data(data), left(nullptr), right(nullptr) {}
};

void preOrderBT(BT *root) {
#if 1
  if (!root)
    return;
  std::stack<BT *> stack;
  stack.push(root);
  while (stack.size()) {
    BT *curr = stack.top();
    stack.pop();
    std::cout << curr->data << " ";
    if (curr->right)
      stack.push(curr->right);
    if (curr->left)
      stack.push(curr->left);
  }
#endif
}

void inOrderBT(BT *root) {
#if 1
  if (!root)
    return;
  std::set<BT *> visited;
  std::stack<BT *> stack;
  stack.push(root);
  while (stack.size()) {
    BT *curr = stack.top();
    stack.pop();
    if (visited.find(curr) != visited.end()) {
      std::cout << curr->data << " ";
      continue;
    }
    visited.insert(curr);

    if (curr->right)
      stack.push(curr->right);

    stack.push(curr);

    if (curr->left)
      stack.push(curr->left);
  }
#endif
}

void postOrderBT(BT *root) {
#if 1
  if (!root)
    return;
  std::set<BT *> visited;
  std::stack<BT *> stack;
  stack.push(root);
  while (stack.size()) {
    BT *curr = stack.top();
    stack.pop();
    if (visited.find(curr) != visited.end()) {
      std::cout << curr->data << " ";
      continue;
    }
    visited.insert(curr);

    stack.push(curr);
    if (curr->right)
      stack.push(curr->right);
    if (curr->left)
      stack.push(curr->left);
  }
#endif
}

struct Node {
  int data;
  std::vector<Node *> neighbors;
  Node(int data) : data(data) {}
};

void DFS(Node *graph) {
  if (!graph)
    return;
  std::set<Node *> visited;
  std::stack<Node *> stack;
  stack.push(graph);
  while (stack.size()) {
    Node *curr = stack.top();
    stack.pop();
    if (visited.count(curr)) {
      std::cout << curr->data << " ";
      continue;
    }

    visited.insert(curr);
    stack.push(curr);

    for (auto neighbor : curr->neighbors)
      if (neighbor && !visited.count(neighbor))
        stack.push(neighbor);
  }
}

// 4.1
bool DFS2(Node *graph, const std::function<bool(Node *)> &test) {
  if (!graph)
    return false;
  std::set<Node *> visited;
  std::stack<Node *> stack;
  stack.push(graph);
  while (stack.size()) {
    Node *curr = stack.top();
    stack.pop();

    if (test(curr))
      return true;

    if (visited.count(curr))
      continue;

    visited.insert(curr);
    stack.push(curr);

    for (auto neighbor : curr->neighbors)
      if (neighbor && !visited.count(neighbor))
        stack.push(neighbor);
  }

  return false;
}

// 4.2
BT *minTree(const std::vector<int> &a) {

  std::vector<BT *> nodes;
  for (auto e : a)
    nodes.push_back(new BT(e));

  std::stack<std::tuple<unsigned, unsigned>> stack;
  stack.push(std::tuple<unsigned, unsigned>(0, a.size() - 1));
  while (stack.size()) {
    auto range = stack.top();
    stack.pop();

    if (!((std::get<1>(range) - std::get<0>(range)) / 2))
      continue;

    unsigned i =
        ((std::get<1>(range) - std::get<0>(range)) / 2) + std::get<0>(range);
    unsigned l = ((i - std::get<0>(range)) / 2) + std::get<0>(range);
    unsigned r = ((std::get<1>(range) - i) / 2) + i;

    std::cout << "i: " << i << "\n";

    if (i == 0 || i >= (a.size() - 1))
      continue;

    BT *root = nodes[i];
    BT *left = nodes[l];
    BT *right = nodes[r];
    root->left = left;
    root->right = right;

    stack.push(std::tuple<unsigned, unsigned>(std::get<0>(range), i));
    stack.push(std::tuple<unsigned, unsigned>(i, std::get<1>(range)));
  }

  return nodes[nodes.size() / 2];
}

// 4.3
std::vector<std::vector<BT *>> getDepths(BT *root) {
  if (!root)
    return {};

  std::map<unsigned, std::vector<BT *>> depthMap;
  depthMap[0] = {root};
  for (unsigned depth = 1; true; depth++) {
    std::vector<BT *> depthNodes;
    for (auto aboveNode : depthMap[depth - 1]) {
      if (aboveNode->left)
        depthNodes.push_back(aboveNode->left);
      if (aboveNode->right)
        depthNodes.push_back(aboveNode->right);
    }

    if (!depthNodes.size())
      break;
    depthMap[depth] = depthNodes;
  }

  std::vector<std::vector<BT *>> result;
  result.resize(depthMap.size());
  for (unsigned i = 0; i < depthMap.size(); i++)
    result[i] = depthMap[i];

  return result;
}

// 4.4
void getDepth(BT *root, std::unordered_map<BT *, unsigned> &depthMap) {
  if (!root)
    return;

  std::stack<BT *> stack;
  stack.push(root);
  std::set<BT *> visited;

  while (stack.size()) {
    BT *curr = stack.top();
    stack.pop();
    if (visited.count(curr)) {
      int ldepth = curr->left ? depthMap[curr->left] : 0;
      int rdepth = curr->right ? depthMap[curr->right] : 0;
      unsigned depth = std::max(ldepth, rdepth);
      depthMap[curr] = depth + 1;
      continue;
    }

    stack.push(curr);
    visited.insert(curr);

    if (curr->left)
      stack.push(curr->left);

    if (curr->right)
      stack.push(curr->right);
  }
}

bool isTreeBalanced(BT *root) {
  if (!root)
    return true;

  std::unordered_map<BT *, unsigned> depthMap;

  getDepth(root, depthMap);

  for (auto entry : depthMap) {
    auto curr = entry.first;
    int ldepth = curr->left ? depthMap[curr->left] : 0;
    int rdepth = curr->right ? depthMap[curr->right] : 0;
    int delta = std::abs(ldepth - rdepth);
    if (delta > 1)
      return false;
  }

  return true;
}

int main() {
  printf("hello\n");
  std::string in;
  std::string in2;
  std::cin >> in;
  std::cin >> in2;
  std::string url = "hello world";
  std::cout << "isUnique: " << isUnique(in) << "\n";
  std::cout << "Repeats: " << hasUniqueChars(in) << "\n";
  std::cout << "Repeats (2): " << hasUniqueChars_2(in) << "\n";
  std::cout << "isPermutation: " << isPermutation(in, in2) << "\n";
  std::cout << "isPermutation_2: " << isPermutation_2(in, in2) << "\n";
  std::cout << "url: " << URLify(url) << "\n";
  std::cout << "isPalPerm " << isPalidromePermutation(in) << "\n";
  std::cout << "edit dist " << editDistance(in, in2) << "\n";
  std::cout << "compress " << compress(in) << "\n";
  std::cout << "decompress " << decompress(compress(in)) << "\n";

  std::vector<int> list = {2, 3, 4, 3, 2, 7, 9, 1, 1};
  LL *head = nullptr;
  for (auto a : list)
    head = new LL(a, head);

  printLL(head);
  dedup(head);
  printLL(head);
  std::cout << "1st last: " << kthLast(head, 1) << "\n";
  std::cout << "2nd last: " << kthLast(head, 2) << "\n";
  std::cout << "3rd last: " << kthLast(head, 3) << "\n";
  std::cout << "4th last: " << kthLast(head, 4) << "\n";
  std::cout << "5th last: " << kthLast(head, 5) << "\n";
  std::cout << "6th last: " << kthLast(head, 6) << "\n";
  std::cout << "7th last: " << kthLast(head, 7) << "\n";
  std::cout << "8th last: " << kthLast(head, 8) << "\n";

  LL *rem = head->next->next;
  std::cout << "Removing: [" << rem->data << "]\n";
  removeNode(head, rem);
  printLL(head);

  int mat[4][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 0, 3, 1}, {1, 2, 3, 4}};

  printMat(mat);
  rotateMat(mat);
  printMat(mat);
  rotateMat(mat);
  printMat(mat);
  rotateMat(mat);
  printMat(mat);
  rotateMat(mat);
  printMat(mat);
  zeroRowCols<4, 4>(mat);
  printMat(mat);
  rotateMat(mat);
  printMat(mat);
  printMat(mat);
  printMat(mat);

  std::cout << "sum list:\n";
  printLL(head);
  std::cout << "sum little: " << sumListLittleEndian_toInt(head, nullptr)
            << "\n";
  std::cout << "sum big: " << sumListBigEndian_toInt(head, nullptr) << "\n";

  std::cout << "sum little to list: ";
  printLL(int_toSumListLittleEndian(sumListLittleEndian_toInt(head, nullptr)));
  std::cout << "sum big: to list: ";
  printLL(int_toSumListBigEndian(sumListBigEndian_toInt(head, nullptr)));
  std::cout << "\n";

  std::vector<int> list2 = {2, 3, 4, 3, 2, 3, 4, 3, 2};
  LL *palLL = nullptr;
  for (auto a : list2)
    palLL = new LL(a, palLL);

  printLL(palLL);
  std::cout << "isPalidrome: " << isLLPalindrome(palLL) << "\n";
  printLL(palLL);

  LL foo(23, palLL);
  std::cout << "Intersecting??\n";
  printLL(palLL);
  std::cout << "\n";
  printLL(&foo);
  std::cout << "\n";
  std::cout << "Intersecting at: ";
  printLL(listIntersect(palLL, &foo));
  std::cout << "\n";

  std::cout << "\n";

  /*
                            1
                          /   \
                         2      5
                        / \    / \
                       3   4  6   7
   */
  BT *root = new BT(1, new BT(2, new BT(3), new BT(4)),
                    new BT(5, new BT(6), new BT(7)));
  std::cout << "pre-order:\n";
  preOrderBT(root);
  std::cout << "\n";
  std::cout << "in-order:\n";
  inOrderBT(root);
  std::cout << "\n";
  std::cout << "post-order:\n";
  postOrderBT(root);
  std::cout << "\n";

  Node *A = new Node(1);
  Node *B = new Node(2);
  Node *C = new Node(3);
  Node *D = new Node(4);
  Node *F = new Node(6);

  // 3 4 2 1
  // C D B A

  A->neighbors.push_back(B);
  A->neighbors.push_back(C);
  B->neighbors.push_back(D);
  D->neighbors.push_back(C);

  // A -> B -> D
  //   -> C <-/

  std::cout << "Graph DFS: \n";
  DFS(A);
  std::cout << "\n";

  std::cout << "Graph Contains A: " << DFS2(A, [&](Node *V) { return A == V; })
            << "\n";
  std::cout << "Graph Contains B: " << DFS2(A, [&](Node *V) { return B == V; })
            << "\n";
  std::cout << "Graph Contains C: " << DFS2(A, [&](Node *V) { return C == V; })
            << "\n";
  std::cout << "Graph Contains F: " << DFS2(A, [&](Node *V) { return F == V; })
            << "\n";

  std::cout << "\n";
  std::cout << "Construct Tree:\n";
  inOrderBT(minTree({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));

  std::cout << "\n";
  std::cout << "Depth Map\n";
  for (auto depth : getDepths(root)) {
    for (auto node : depth)
      std::cout << node->data << " ";
    std::cout << "\n";
  }

  std::unordered_map<BT *, unsigned> depthMap;

  getDepth(root, depthMap);

  for (auto entry : depthMap) {
    auto curr = entry.first;
    int ldepth = curr->left ? depthMap[curr->left] : 0;
    int rdepth = curr->right ? depthMap[curr->right] : 0;
    std::cout << " curr: " << curr->data << " depths: " << ldepth << " "
              << rdepth << "\n";
  }

  std::cout << "\n";
  std::cout << "\n";
}
