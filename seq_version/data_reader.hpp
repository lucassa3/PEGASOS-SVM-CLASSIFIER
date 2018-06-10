#ifndef DATA_READER_H
#define DATA_READER_H

#include <vector>
#include <iostream>
#include <string>

using namespace std;

vector<vector<double>> read_data(const char * filename, const unsigned int data_length);

vector<double> set_labels(vector<vector<double>> & data);

int feature_size(vector<vector<double>> & data);

#endif