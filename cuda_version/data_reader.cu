#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <boost/algorithm/string.hpp>

#include "data_reader.hpp"

using namespace std;

vector<vector<double>> read_data(const char * filename, const unsigned int data_length) {
    
    ifstream file(filename);
 
	vector<vector<double>> data;
 
	string line = "";
    unsigned int counter = 0;


	while (getline(file, line) && counter < data_length) 
    {
		vector<string> vec;
        vector<double> dvec;

		boost::algorithm::split(vec, line, boost::is_any_of(","));

        for(unsigned int i = 0; i < vec.size(); i++) {

            if(vec[i] == "Iris-setosa") {
                dvec.push_back(1);
            }

            else if(vec[i] == "Iris-versicolor") {
                dvec.push_back(0);
            }

            else if(vec[i] == "Iris-virginica") {
                dvec.push_back(2);
            }

            else

            dvec.push_back(atof(vec[i].c_str()));

            
        }

        data.push_back(dvec);
        counter++;
	}

	file.close();

    return data;
}

vector<double> set_labels(vector<vector<double>> & data) {
    
    vector<double> labels;

    double total = 0;
    
    for (unsigned int i = 0; i < data.size(); i++) {
        
        if (data[i][data[i].size()-1] == 1) {
            labels.push_back(1);
            total+=1;

        }

        else labels.push_back(-1);
    }
    cout << "distribuicao: " << total/data.size() << endl;
    cout << data[0].size()-1 << endl;


    //replace data class row for bias
    for (unsigned int i = 0; i < data.size(); i++) {
        
        data[i][data[0].size()-1] = 1;
    }

    return labels;
}

int feature_size(vector<vector<double>> & data) {
    return data[0].size();
}