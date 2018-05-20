CFLAGS = -std=c++11 -Wall

all: main.o svm_classifier.o
	g++ $(CFLAGS) $^ -o main

main.o: main.cpp svm_classifier.hpp
	g++ $(CFLAGS) -c main.cpp

svm_classifier.o: svm_classifier.cpp svm_classifier.hpp
	g++ $(CFLAGS) -c svm_classifier.cpp

clean:
	rm -rf *.o main