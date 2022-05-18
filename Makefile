all:
	clang++ main.cpp -I/usr/include/tensorflow -ltensorflow_cc -ltensorflow_framework -lhiredis -o worker
run: all
	./a.out
