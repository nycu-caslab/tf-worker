all:
	clang++ main.cpp -I/usr/include/tensorflow -ltensorflow_cc -ltensorflow_framework
run: all
	./a.out
