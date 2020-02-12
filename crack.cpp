#include <algorithm>
#include <bitset>
#include <cstdio>
#include <iostream>
#include <numeric>
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
    for (unsigned si = 0, li = 0; li < ls.size(); li++)
      ((si > ss.size() || ls[li] != ss[si]) ? edits : si)++;
    return edits;
  }();
}

int main() {
  printf("hello\n");
  std::string in;
  std::string in2;
  std::cin >> in;
  std::cin >> in2;
  std::string url = "fuck you";
  std::cout << "Repeats: " << hasUniqueChars(in) << "\n";
  std::cout << "Repeats (2): " << hasUniqueChars_2(in) << "\n";
  std::cout << "isPermutation: " << isPermutation(in, in2) << "\n";
  std::cout << "isPermutation_2: " << isPermutation_2(in, in2) << "\n";
  std::cout << "url: " << URLify(url) << "\n";
  std::cout << "isPalPerm " << isPalidromePermutation(in) << "\n";
  std::cout << "edit dist " << editDistance(in, in2) << "\n";
}
