#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <string>
#include <time.h>
#include <algorithm>
#include "svm_classifier.hpp"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>

using namespace std;

double rand_float(double low, double high) {
    
    return ((double)rand() * (high - low)) / (double)RAND_MAX + low;
}

vector<vector<double>> read_data(const char * filename, const unsigned int data_length, const unsigned int jump_lines) {
    
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
            
            /*if(i == vec.size()-1) {
                if(vec[i] == "Iris-setosa") {
                    dvec.push_back(1);
                }

                else if(vec[i] == "Iris-versicolor") {
                    dvec.push_back(0);
                }

                else if(vec[i] == "Iris-virginica") {
                    dvec.push_back(2);
                }
            }

            else*/

            dvec.push_back(atof(vec[i].c_str()));

            
        }

        /*for(unsigned int i = 0; i < dvec.size(); i++) { 
            cout << dvec[i] << " ";
        }

        cout << endl;*/
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
    cout << data[0].size()-1 << endl; //8


    //replace data class row for bias
    for (unsigned int i = 0; i < data.size(); i++) {
        
        data[i][data[0].size()-1] = 1;
    }

    return labels;
}


int main(int argc, char *argv[]) {
    
    vector<vector<double>> h_data = read_data("../datasets/diabetes.csv", 5000000,  0);
    
    random_shuffle(h_data.begin(), h_data.end());

    vector<double> h_labels = set_labels(h_data);
    
    

    // for(unsigned int i = 0; i < dataset.training_images.size(); i++) {
    //     for(unsigned int j = 0; j < dataset.training_images[i].size(); j++) cout << dataset.training_images[i][j] << " ";
    //     cout << endl;
    // }
    
    
    /*for(unsigned int i = 0; i < data.size(); i++) {
        for(unsigned int j = 0; j < data[i].size(); j++) cout << data[i][j] << " ";
        cout << endl;
    }*/
    thrust::device_vector<double> data;


    // for (unsigned int i = 0; i < h_data.size(); i++) {
        
    //     data[i][data[0].size()-1] = 1;
    // }

    for (vector<double> sample: h_data) {
        thrust::device_vector<double> aux(sample);
        data.insert(data.end(), aux.begin(), aux.end());   
    }

    thrust::copy(data.begin(), data.begin()+9, std::ostream_iterator<int>(std::cout, "\n"));
    

    //thrust::device_vector<thrust::device_vector<double>> data(h_data);
    thrust::device_vector<double> labels(h_labels);
    thrust::copy(labels.begin(), labels.begin()+1, std::ostream_iterator<int>(std::cout, "\n"));
    
    //for(unsigned int j = 0; j < labels.size(); j++) cout << labels[j] << endl;

    

    thrust::device_vector<double> x_train(data.begin(), data.begin() + (data.size()/2) - ((data.size()/2)%9));
    thrust::device_vector<double> x_test(data.begin() + (data.size()/2) - ((data.size()/2)%9), data.end());

    thrust::device_vector<int> y_train(labels.begin(), labels.begin() + labels.size()/2);
    thrust::device_vector<int> y_test(labels.begin() + labels.size()/2, labels.end());
    cout << x_train.size() << endl;
    cout << y_train.size() << endl;



    double total_acc = 0;

    for (unsigned int i = 0; i < 20; i++) {

        SVMClassifier* svm_clf = new SVMClassifier(0.001, 1000000, time(NULL));
        cout << "seed: " << time(NULL) << endl;

        svm_clf->fit(x_train, y_train);

        thrust::device_vector<int> y_pred = svm_clf->predict(x_test);

        double cur_acc = svm_clf->accuracy(y_test, y_pred);

        //for(unsigned int j = 0; j < y_pred.size(); j++) cout << y_pred[j] << endl;

        total_acc += cur_acc;

        cout << "accurarcy: "<< cur_acc  << endl;
    }

    cout << "mean accuracy: "<< total_acc/20  << endl;

    return 0;
}