#ifndef UTILS_HPP
#define UTILS_HPP

#include <fstream>
#include <iostream>

template <typename Arg, typename... Args> void LOGs(Arg &&arg, Args &&...args) {
  std::ofstream stream;
  stream.open("log.txt", std::ofstream::out | std::ofstream::app);
  stream << std::forward<Arg>(arg);
  using expander = int[];
  (void)expander{0, (void(stream << ' ' << std::forward<Args>(args)), 0)...};
  stream << '\n';
}

#endif
