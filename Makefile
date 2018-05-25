CFLAGS = -std=c++11 -Wall -I ../mnist-master/include/mnist

all: main.o svm_classifier.o data_reader.o
	g++ $(CFLAGS) $^ -o main 

main.o: main.cpp svm_classifier.hpp data_reader.hpp
	g++ $(CFLAGS) -c main.cpp

svm_classifier.o: svm_classifier.cpp svm_classifier.hpp
	g++ $(CFLAGS) -c svm_classifier.cpp

data_reader.o: data_reader.cpp data_reader.hpp
	g++ $(CFLAGS) -c data_reader.cpp

clean:
	rm -rf *.o main