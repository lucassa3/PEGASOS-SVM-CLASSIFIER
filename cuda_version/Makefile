CFLAGS = -std=c++11

all: main.o svm_classifier.o data_reader.o
	nvcc $(CFLAGS) $^ -o main

main.o: main.cu svm_classifier.hpp data_reader.hpp
	nvcc $(CFLAGS) -c main.cu

svm_classifier.o: svm_classifier.cu svm_classifier.hpp
	nvcc $(CFLAGS) -c svm_classifier.cu

data_reader.o: data_reader.cu data_reader.hpp
	nvcc $(CFLAGS) -c data_reader.cu

clean:
	rm -rf *.o main