#ifndef UTILS_HPP
#define UTILS_HPP

#include <fstream>
#include <iostream>

#define LOGs(...) log2file(__LINE__, __FILE__, __VA_ARGS__)

template <typename... Args>
void log2file(int line, const char *fileName, Args &&...args) {
  std::ofstream stream;
  stream.open("tf-worker/log.txt", std::ofstream::out | std::ofstream::app);
  stream << fileName << "(" << line << ") : ";
  (stream << ... << std::forward<Args>(args)) << '\n';
}

#endif
