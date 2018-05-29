#ifndef DATA_READER_H
#define DATA_READER_H

#include <vector>
#include <iostream>
#include <string>
#include <thrust/host_vector.h>

using namespace std;

thrust::host_vector<thrust::host_vector<double>> read_data(const char * filename, const unsigned int data_length, const unsigned int jump_lines);

thrust::host_vector<double> set_labels(thrust::host_vector<thrust::host_vector<double>> & data);

#endif