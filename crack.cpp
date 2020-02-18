#include <algorithm>
#include <bitset>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <vector>
#include <memory>

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

bool isPalidromePermutation(const std::string &str) {
  std::bitset<256> bits;
  for (auto c : str)
    bits.flip(c);
  return bits.count() <= 1;
}

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

struct LL {
  int data;
  LL *next;
  LL(int data, LL *next) : data(data), next(next) {}
};

template <typename T> struct LL2 {
  T data;
  std::unique_ptr<LL2<T>> next;
  LL2(T data) : data(data), next(nullptr) {}
  LL2(T data, std::unique_ptr<LL2<T>> next)
      : data(data), next(std::move(next)) {}
};

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

int kthLast(LL *head, int k) {
  unsigned count = 0;
  for (LL *curr = head; curr; curr = curr->next)
    count++;
  for (LL *curr = head; curr; curr = curr->next)
    if ((count-- - k) <= 0)
      return curr->data;
  return head->data;
}

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

int main() {
  printf("hello\n");
  std::string in;
  std::string in2;
  std::cin >> in;
  std::cin >> in2;
  std::string url = "hello world";
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
  auto printLL = [](const LL *head) {
    for (const LL *curr = head; curr; curr = curr->next)
      std::cout << "[" << curr->data << "]->";
    std::cout << "//\n";
  };
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

  std::cout << "\n";

}
