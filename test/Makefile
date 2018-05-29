CFLAGS = -std=c++11

all: main.o 
	nvcc $(CFLAGS) $^ -o main 

main.o: main.cpp
	nvcc $(CFLAGS) -c main.cpp

clean:
	rm -rf *.o main