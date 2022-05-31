#include <fstream>
int main() {

  std::ofstream log;
  log.open("log2");

  log << "Process start\n";
  log.close();
}
