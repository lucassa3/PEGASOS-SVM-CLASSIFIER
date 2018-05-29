CFLAGS = -std=c++11

all: main.o 
	nvcc $(CFLAGS) $^ -o main 

main.o: main.cu
	nvcc $(CFLAGS) -c main.cu

clean:
	rm -rf *.o main