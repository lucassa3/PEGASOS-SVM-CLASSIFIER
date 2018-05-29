CFLAGS = -std=c++11

all: test.o svm_classifier.o
	nvcc $(CFLAGS) $^ -o test 

test.o: test.cu svm_classifier.hpp
	nvcc $(CFLAGS) -c test.cu

svm_classifier.o: svm_classifier.cu svm_classifier.hpp
	nvcc $(CFLAGS) -c svm_classifier.cu

clean:
	rm -rf *.o test