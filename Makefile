all:
	clang++ -std=c++17 main.cpp -I/usr/include/tensorflow -ltensorflow_cc -ltensorflow_framework -lhiredis -o worker -O2
docker:
	g++ -std=c++14 main.cpp -I/usr/local/include/tensorflow/bazel-bin/tensorflow/include -ltensorflow_cc -lhiredis -o /worker -O2
run: all
	./a.out
