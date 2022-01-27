all:
	clang++ main.cpp -I/usr/include/tensorflow -ltensorflow_cc -ltensorflow_framework -lhiredis
run: all
	./a.out
