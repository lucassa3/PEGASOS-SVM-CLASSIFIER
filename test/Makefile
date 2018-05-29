CFLAGS = -std=c++11

all: test.o 
	nvcc $(CFLAGS) $^ -o test 

test.o: test.cu
	nvcc $(CFLAGS) -c test.cu

clean:
	rm -rf *.o test