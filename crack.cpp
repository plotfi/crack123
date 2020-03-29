#include <algorithm>
#include <arpa/inet.h>
#include <bitset>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <stack>
#include <unordered_map>
#include <unordered_set>
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
  for (const auto &c : str)
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
  for (const auto &c : sortedStr) {
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
  for (const auto &c : str)
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
  for (const auto &c : a)
    charCountA[c]++;

  std::vector<size_t> charCountB;
  charCountB.resize(256, 0);
  for (const auto &c : b)
    charCountB[c]++;

  return std::equal(charCountA.begin(), charCountA.end(), charCountB.data());
}

bool isPermutation_2(const std::string &a, const std::string &b) {
  std::vector<int> count;
  count.resize(256, 0);
  for (const auto &c : a)
    count[c]++;

  for (const auto &c : b)
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
  for (const auto &c : str)
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
  for (const auto &c : str) {
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

template <typename T, unsigned N> void printMatT(T matrix[N][N]) {
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

  for (const auto &row : rows)
    for (unsigned j = 0; j < N; j++)
      matrix[row][j] = 0;
  for (const auto &col : cols)
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
  BT *parent;
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

template <typename T> struct NodeT1 {
  T data;
  std::set<NodeT1<T> *> neighbors;
  NodeT1(T data) : data(data) {}
};

using Node = NodeT1<int>;

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

    for (const auto &neighbor : curr->neighbors)
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

    for (const auto &neighbor : curr->neighbors)
      if (neighbor && !visited.count(neighbor))
        stack.push(neighbor);
  }

  return false;
}

// 4.2
BT *minTree(const std::vector<int> &a) {

  std::vector<BT *> nodes;
  for (const auto &e : a)
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
    for (const auto &aboveNode : depthMap[depth - 1]) {
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

  for (const auto &entry : depthMap) {
    auto curr = entry.first;
    int ldepth = curr->left ? depthMap[curr->left] : 0;
    int rdepth = curr->right ? depthMap[curr->right] : 0;
    int delta = std::abs(ldepth - rdepth);
    if (delta > 1)
      return false;
  }

  return true;
}

// 4.5
bool isBST(BT *root) {
  if (!root)
    return true;

  std::stack<BT *> stack;
  stack.push(root);
  while (stack.size()) {
    BT *curr = stack.top();
    stack.pop();

    if (curr->right) {
      if (curr->right->data < curr->data)
        return false;
      stack.push(curr->right);
    }
    if (curr->left) {
      if (curr->left->data > curr->data)
        return false;
      stack.push(curr->left);
    }
  }

  return true;
}

// 4.6
BT *getNextInOrderNode(BT *node) {
  if (!node || !node->right)
    while (true) {
      if (!node || !node->parent)
        return nullptr;
      if (node->parent->left == node)
        return node->parent;
      node = node->parent;
    }

  for (node = node->right; node->left; node = node->left)
    ;

  return node;
}

template <typename T> struct NodeT {
  T data;
  std::set<NodeT<T> *> neighbors;
  NodeT(T data) : data(data) {}
};

// 4.7: reverse post order
void printDependencyOrder(const std::vector<char> &jobs,
                          const std::vector<std::tuple<char, char>> &deps) {
  using NodeC = NodeT<char>;
  std::unordered_map<char, std::unique_ptr<NodeC>> NodeMap;
  std::unordered_set<char> roots;

  // Allocate Jobs Nodes and build root list (start with all).
  for (const auto &j : jobs) {
    if (NodeMap.find(j) == NodeMap.end())
      NodeMap[j] = std::make_unique<NodeC>(j);
    roots.insert(j);
  }

  const size_t totalDedupedJobs = roots.size();
  // build dependecy graph and prune the roots list.
  for (const auto &dep : deps) {
    auto S = NodeMap[std::get<0>(dep)].get();
    auto D = NodeMap[std::get<1>(dep)].get();
    auto II = roots.find(D->data);
    if (II != roots.end())
      roots.erase(II);
    S->neighbors.insert(D);
  }

  using Expected = std::tuple<std::vector<char>, std::string, bool>;
  auto getRPO = [totalDedupedJobs](
                    const std::unordered_set<char> &roots,
                    std::unordered_map<char, std::unique_ptr<NodeC>> &NodeMap)
      -> Expected {
    std::vector<char> RPO;

    // DFS the roots.
    std::stack<NodeC *> stack;
    std::map<NodeC *, unsigned> visited;
    NodeC TheRoot('\0');
    stack.push(&TheRoot);
    for (const auto &root : roots)
      TheRoot.neighbors.insert(NodeMap[root].get());
    while (stack.size()) {
      auto C = stack.top();
      stack.pop();
      if (visited.count(C)) {
        visited[C]++;
        RPO.push_back(C->data);
        continue;
      }

      visited[C] = 0;
      stack.push(C);
      for (const auto &neighbor : C->neighbors) {
        if (!neighbor)
          continue;
        auto II = visited.find(neighbor);
        if (II != visited.end() && II->second == 0) {
          std::cout << "Cycle: " << C->data << " " << neighbor->data << "\n";
          return {RPO, "Encounted Cycle", false};
        }

        if (!visited.count(neighbor))
          stack.push(neighbor);
      }
    }

    RPO.pop_back();
    if (RPO.size() != totalDedupedJobs)
      return {RPO, "Encounted Cycle", false};

    std::reverse(RPO.begin(), RPO.end());
    return {RPO, "SUCCESS", true};
  };

  auto [RPO, RPOErrMsg, RPOErr] = getRPO(roots, NodeMap);
  if (!RPOErr) {
    std::cout << "Invalid Schedule: " << RPOErrMsg << "\n";
    return;
  }

  std::cout << "RPO: ";
  for (const auto &c : RPO)
    std::cout << c << " ";
  std::cout << "\n";
}

// 4.8 first common ancestor
BT *firstCommonAncestor(BT *root, BT *node1, BT *node2) {
  auto populateParentMap = [](BT *root) {
    std::unordered_map<BT *, BT *> ParentMap;
    std::stack<BT *> stack;
    std::set<BT *> visited;
    stack.push(root);
    ParentMap[root] = nullptr;
    while (stack.size()) {
      auto C = stack.top();
      stack.pop();
      if (visited.count(C))
        continue;

      visited.insert(C);
      stack.push(C);
      if (C->left) {
        stack.push(C->left);
        ParentMap[C->left] = C;
      }
      if (C->right) {
        stack.push(C->right);
        ParentMap[C->right] = C;
      }
    }

    return ParentMap;
  };

  auto depth = [](BT *node, std::unordered_map<BT *, BT *> &ParentMap) {
    size_t depth = 0;
    for (BT *curr = node; curr; curr = ParentMap[curr])
      depth++;
    return depth;
  };

  auto getCommonAncestor =
      [](BT *node1, BT *node2, size_t depth1, size_t depth2,
         std::unordered_map<BT *, BT *> &ParentMap) -> BT * {
    if (depth1 != depth2)
      for (unsigned i = 0; i < depth1 - depth2; i++) {
        if (!node1)
          break;
        node1 = ParentMap[node1];
      }

    while (node1 && node2) {
      if (node1 == node2)
        return node1;
      node1 = ParentMap[node1];
      node2 = ParentMap[node2];
    }
    return nullptr;
  };

  if (!root || !node1 || !node2)
    return nullptr;

  std::unordered_map<BT *, BT *> ParentMap = populateParentMap(root);
  size_t depth1 = depth(node1, ParentMap);
  size_t depth2 = depth(node2, ParentMap);

  if (!depth1 || !depth2)
    return nullptr;

  return depth1 < depth2
             ? getCommonAncestor(node2, node1, depth2, depth1, ParentMap)
             : getCommonAncestor(node1, node2, depth1, depth2, ParentMap);
}

// 5.1: set bits
unsigned setBits(unsigned N, unsigned M, unsigned i, unsigned j) {
  const unsigned mask = (1 << (j - i + 1)) - 1;
  return (N & (~mask << i)) | ((mask & M) << i);
}

// 5.2 Binary To String

// 5.3: get longest bit range
unsigned getLongest(unsigned n) {

  std::vector<std::tuple<unsigned, unsigned>> ranges;
  std::unordered_map<unsigned, std::tuple<unsigned, unsigned>> HiRangeMap;
  std::unordered_map<unsigned, std::tuple<unsigned, unsigned>> LoRangeMap;

  unsigned curr = 0;
  unsigned last = 0;
  bool lastBitWasOne = false;
  for (unsigned I = 1, E = 1 << 31; I != E; I <<= 1) {
    if (n & I) {
      if (!lastBitWasOne)
        last = curr;

      lastBitWasOne = true;
    } else {
      if (lastBitWasOne) {
        ranges.push_back({last, curr - 1});
        HiRangeMap[std::get<1>(ranges.back())] = ranges.back();
        LoRangeMap[std::get<0>(ranges.back())] = ranges.back();
      }
      lastBitWasOne = false;
    }

    curr++;
  }

  unsigned max = 0;
  for (const auto &range : ranges) {

    size_t rangeSize = std::get<1>(range) - std::get<0>(range) + 1;
    if (rangeSize > max)
      max = rangeSize;

    if (std::get<0>(range) > 0 || std::get<1>(range) < 31)
      if (rangeSize + 1 > max)
        max = rangeSize + 1;

    std::cout << "range: " << std::get<0>(range) << " - " << std::get<1>(range)
              << "\n";

    if (std::get<0>(range) > 1) {
      auto II = HiRangeMap.find(std::get<0>(range) - 2);
      if (HiRangeMap.end() != II) {
        auto sideRange = II->second;
        size_t newRangeSize =
            (rangeSize + 2 + (std::get<1>(sideRange) - std::get<0>(sideRange)));
        if (newRangeSize > max)
          max = newRangeSize;
      }
    }

    if (std::get<1>(range) < 30) {
      auto II = LoRangeMap.find(std::get<1>(range) + 2);
      if (LoRangeMap.end() != II) {
        auto sideRange = II->second;
        size_t newRangeSize =
            (rangeSize + 2 + (std::get<1>(sideRange) - std::get<0>(sideRange)));
        if (newRangeSize > max)
          max = newRangeSize;
      }
    }
  }

  std::cout << "max: " << max << "\n";
  return max;
}

// 5.4  Next Number: Given a positive integer, print the next smallest and the
// next largest number that have the same number of 1 bits in their binary
// representation.

// 5.5 Debugger: Explain what the following code does: ( (n & (n - 1) ) e) .

// 5.6 Conversion: Write a function to determine the number of bits you would
// need to flip to convert integer A to integer B.

// 5.7 Pairwise Swap: Write a program to swap odd and even bits in an integer
// with as few instructions as possible (e.g., bit 13 and bit 1 are swapped,
// bit 2 and bit 3 are swapped, and so on).

// 5.8 Draw Line

// find holes
// find 2 holes, 3 holes?
// 4.9: Horrible permutation problem

// 4.10 find sub tree in a larger tree (GVN?)

// 4.11 Random Node from a binary tree

// 4.12 Path with sum

// find holes
// find 2 holes, 3 holes?

// TODO:
// 4.9: BST Sequence
// 4.10: Check Subtree
// 4.11: Random Node
// 4.12: Path With Sums
// 5.2: real number to string
// BFS

// Dynamic Programming and Recursion

unsigned superFib(unsigned n) {
  if (n < 2)
    return n;

  unsigned b = 1;
  for (unsigned i = 0, a = 0; i < n; i++, b += a)
    std::swap(a, b);
  return b;
}

unsigned fibNonRecursive(unsigned n) {
  if (n < 2)
    return n;

  std::vector<unsigned> memo;
  memo.resize(n + 1, 0);
  memo[1] = 1;

  for (unsigned i = 2; i < memo.size(); i++)
    memo[i] = memo[i - 1] + memo[i - 2];

  return memo.back();
}

unsigned fibonacci(unsigned n, std::vector<unsigned> &memo) {
  if (memo[n] != 0)
    return memo[n];

  std::cout << "Have to calculate fib(" << n << ")\n";

  unsigned fibn = n < 2 ? n : fibonacci(n - 1, memo) + fibonacci(n - 2, memo);
  memo[n] = fibn;
  return fibn;
}

unsigned fibonacci(unsigned n) {
  std::vector<unsigned> memo;
  memo.resize(n + 1, 0);
  return fibonacci(n, memo);
}

// 8.1
unsigned steps(unsigned n) {
  if (n < 3)
    return n;
  if (n == 3)
    return 4;

  std::vector<unsigned> memo;
  memo.resize(n + 1, 0);
  memo[0] = 0;
  memo[1] = 1;
  memo[2] = 2;
  memo[3] = 4;

  for (unsigned i = 4; i < n + 1; i++)
    memo[i] = memo[i - 1] + 1 + !(i % 2) + !(i % 3);

  return memo.back();
}

// 8.2 Robot
template <unsigned N, unsigned M> unsigned findPath(const bool Grid[N][M]) {

  unsigned PathGrid[N][M];
  bzero(PathGrid, sizeof(PathGrid));

  for (unsigned j = 0; j < M; j++)
    PathGrid[0][j] = Grid[0][j] ? j : ~0U;

  for (unsigned i = 1; i < N; i++)
    for (unsigned j = 0; j < M; j++) {
      unsigned min = (j >= i && Grid[i][j])
                         ? std::min(PathGrid[i - 1][j], PathGrid[i][j - 1])
                         : ~0U;
      PathGrid[i][j] = min == ~0U ? ~0U : min + 1;
    }

  printMatT(PathGrid);

  return PathGrid[N - 1][M - 1];
}

// 8.3 magic

// 8.4 powerset
std::vector<std::string> printPowerSet(const std::set<char> &set) {
  std::vector<char> chars;
  std::transform(set.begin(), set.end(), std::back_inserter(chars),
                 [](const char c) { return c; });

  std::vector<std::string> sets;
  sets.resize(1 << set.size());

  for (uint64_t I = 0, E = sets.size(); I != E; I++)
    for (uint64_t i = I, count = 0; i; i >>= 1, count++)
      if (1 & i)
        sets[I] += chars[count];

  return sets;
}

// 8.5 product with no multiply operator
unsigned product(unsigned a, unsigned b) {
  unsigned product = 0;

  auto popcount = [](unsigned c) {
    return std::bitset<sizeof(c) * CHAR_BIT>(c).count();
  };

  while (a && b) {
    unsigned sig = b;
    while (unsigned next = (sig & (sig - 1)))
      sig = next;
    product += a << popcount(sig - 1);
    b &= sig - 1;
  }

  return product;
}

// 8.6 tower of hanoai wtf

// 8.7

// 8.11 coins: unique ways to make change
unsigned coins(unsigned n) {
  std::vector<unsigned> denoms = {0, 1, 5, 10, 25};
  unsigned memo[denoms.size()][n + 1];

  for (unsigned i = 0; i <= n; i++)
    memo[0][i] = 0;
  for (unsigned i = 0; i < denoms.size(); i++)
    memo[i][0] = 1;

  for (unsigned j = 1; j < denoms.size(); j++)
    for (unsigned i = 1; i <= n; i++)
      if (i < denoms[j])
        memo[j][i] = memo[j - 1][i];
      else
        memo[j][i] = memo[j - 1][i] + memo[j][i - denoms[j]];

  return memo[denoms.size() - 1][n];
}

void *mymemcpy(void *dst, const void *src, size_t n) {

  auto bytecpy = [](void *&dst, const void *&src, size_t n) {
    auto dstPtr = (char *)dst;
    auto srcPtr = (const char *)src;
    while (n--)
      *dstPtr++ = *srcPtr++;
    dst = dstPtr;
    src = srcPtr;
    return dst;
  };

  if (n < sizeof(uint32_t))
    return bytecpy(dst, src, n);

  uintptr_t dstMisAligned = ((uintptr_t)dst) % sizeof(uint32_t);
  uintptr_t srcMisAligned = ((uintptr_t)src) % sizeof(uint32_t);

  if (dstMisAligned == srcMisAligned) {

    if (dstMisAligned)
      bytecpy(dst, src, dstMisAligned);

    const unsigned totalChunks = ((n - dstMisAligned) / sizeof(uint32_t));
    const unsigned rem = n - (totalChunks * sizeof(uint32_t)) - dstMisAligned;

    auto dstPtr = (uint32_t *)dst;
    auto srcPtr = (const uint32_t *)src;
    for (unsigned i = 0; i < totalChunks; i++)
      *dstPtr++ = *srcPtr++;
    dst = dstPtr;
    src = srcPtr;

    if (rem)
      bytecpy(dst, src, rem);

    return dst;
  }

  if (dstMisAligned > srcMisAligned) {

    bytecpy(dst, src, dstMisAligned);

    const unsigned misAlignDelta = dstMisAligned - srcMisAligned;
    const unsigned totalChunks = ((n - dstMisAligned) / sizeof(uint32_t));
    const unsigned rem = n - (totalChunks * sizeof(uint32_t)) - dstMisAligned;

    auto dstPtr = (uint32_t *)dst;
    auto srcPtr = (const uint32_t *)(((const char *)src) - misAlignDelta);
    for (unsigned i = 0; i < totalChunks; i++) {
      *dstPtr = ntohl(htonl(*srcPtr) << (misAlignDelta * 8) |
                      htonl(*(srcPtr + 1)) >>
                          ((sizeof(uint32_t) - misAlignDelta) * 8));
      dstPtr++;
      srcPtr++;
    }
    dst = dstPtr;
    src = srcPtr;
    src = ((char *)src) + misAlignDelta;

    if (rem)
      bytecpy(dst, src, rem);

    return dst;
  }

  if (dstMisAligned < srcMisAligned) {

    bytecpy(dst, src, dstMisAligned + sizeof(uint32_t));

    const unsigned misAlignDelta = srcMisAligned - dstMisAligned;
    const unsigned totalChunks =
        ((n - (dstMisAligned + sizeof(uint32_t))) / sizeof(uint32_t));
    const unsigned rem = n - (totalChunks * sizeof(uint32_t)) -
                         (dstMisAligned + sizeof(uint32_t));

    auto dstPtr = (uint32_t *)dst;
    auto srcPtr = (const uint32_t *)(((const char *)src) + misAlignDelta);
    for (unsigned i = 0; i < totalChunks; i++) {
      uint32_t result1 = htonl(*srcPtr) >> (misAlignDelta * 8) &
                         ((1 << ((sizeof(uint32_t) - misAlignDelta) * 8)) - 1);
      uint32_t result2 = htonl(*(srcPtr - 1))
                         << ((sizeof(uint32_t) - misAlignDelta) * 8);
      *dstPtr++ = ntohl(result1 | result2);
      srcPtr++;
    }
    dst = dstPtr;
    src = srcPtr;
    src = ((char *)src) - misAlignDelta;

    if (rem)
      bytecpy(dst, src, rem);

    return dst;
  }

  return bytecpy(dst, src, n);
}

int main() {
  printf("hello\n");
  std::string in = "foo";
  std::string in2 = "bar";
  // std::cin >> in;
  // std::cin >> in2;
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
  for (const auto &a : list)
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
  for (const auto &a : list2)
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

  root->parent = nullptr;
  root->left->parent = root;
  root->right->parent = root;
  root->left->left->parent = root->left;
  root->left->right->parent = root->left;
  root->right->right->parent = root->right;
  root->right->left->parent = root->right;

  BT *leftmost = root;
  while (leftmost->left)
    leftmost = leftmost->left;

  std::cout << "get next node inorder:\n";
  for (; leftmost; leftmost = getNextInOrderNode(leftmost))
    std::cout << "next inorder: " << leftmost->data << "\n";
  std::cout << "\n";

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

  A->neighbors.insert(B);
  A->neighbors.insert(C);
  B->neighbors.insert(D);
  D->neighbors.insert(C);

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
  for (const auto &depth : getDepths(root)) {
    for (const auto &node : depth)
      std::cout << node->data << " ";
    std::cout << "\n";
  }

  std::unordered_map<BT *, unsigned> depthMap;

  getDepth(root, depthMap);

  for (const auto &entry : depthMap) {
    auto curr = entry.first;
    int ldepth = curr->left ? depthMap[curr->left] : 0;
    int rdepth = curr->right ? depthMap[curr->right] : 0;
    std::cout << " curr: " << curr->data << " depths: " << ldepth << " "
              << rdepth << "\n";
  }

  std::cout << "\n";
  std::cout << "Print Dependency ordering:\n";
  printDependencyOrder({'a', 'b', 'c', 'd', 'e', 'f'}, {
                                                           {'a', 'd'},
                                                           {'f', 'b'},
                                                           {'b', 'd'},
                                                           {'b', 'd'},
                                                           {'f', 'a'},
                                                           {'d', 'c'},
                                                           {'c', 'f'},
                                                       });
  std::cout << "\n";
  std::cout << "\n";
  std::cout << "Print Dependency ordering:\n";
  printDependencyOrder({'a', 'b', 'c', 'd', 'e', 'f'}, {
                                                           {'a', 'd'},
                                                           {'f', 'b'},
                                                           {'b', 'd'},
                                                           {'b', 'd'},
                                                           {'f', 'a'},
                                                           {'d', 'c'},
                                                       });

  std::cout << "Common ancestor: \n";
  std::cout
      << firstCommonAncestor(root, root->left->right, root->left->left)->data;
  std::cout << "\n";

  getLongest(10);
  getLongest(31);
  std::cout << "\n";

  std::cout << "Fib: " << fibonacci(23) << "\n";
  std::cout << "Fib no-rev: " << fibNonRecursive(23) << "\n";
  for (unsigned i = 0; i <= 23; i++)
    std::cout << "Fib super (" << i << "): " << superFib(i) << "\n";

  std::cout << "\n";
  std::cout << "\n";

  std::cout << "steps: " << steps(5) << "\n";

  std::cout << "\n";
  std::cout << "\n";

  bool grid[4][4] = {{true, true, true, true},
                     {true, true, false, true},
                     {true, true, true, true},
                     {true, true, true, true}};

  std::cout << "Find Path: " << findPath<4, 4>(grid) << "\n";

  std::cout << "\n";
  std::cout << "\n";
  std::cout << "print power:\n";
  for (const auto &set : printPowerSet({'a', 'b', 'c'}))
    std::cout << "{" << set << "}\n";
  std::cout << "\n";
  std::cout << "\n";
  std::cout << "product: " << product(15, 35) << " --> " << 15 * 35 << "\n";
  std::cout << "\n";
  std::cout << "\n";
  for (unsigned i = 0; i < 101; i++)
    std::cout << "coins: " << i << " " << coins(i) << "\n";
  std::cout << "\n";
  std::cout << "\n";

  char buffer[1024];
  const char *sofa = "I am sofa king we todd ed...............................";
  char *sofa2 = (char*)sofa;
  sofa2++;
  sofa2++;
  sofa2++;
  sofa2++;
  sofa2++;
  sofa2++;
  sofa2++;
  sofa2++;
  sofa2++;
  sofa2++;
  sofa2++;
  mymemcpy(buffer, sofa2, 26);
  std::cout << "printing message: " << (char *)buffer << "\n";
  std::cout << "\n";
  std::cout << "\n";
  std::cout << "\n";
}
