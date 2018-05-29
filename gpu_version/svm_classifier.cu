#include <random>
#include <vector>
#include <iostream>
#include <string>
#include <time.h>
#include "svm_classifier.hpp"
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>

SVMClassifier::SVMClassifier(double c, unsigned int epochs, unsigned int seed) {
  this->c = c;
  this->epochs = epochs;
  this->seed = seed;
}



void SVMClassifier::fit(thrust::device_vector<thrust::device_vector<double>> & data, thrust::device_vector<int> & label) {
    srand(seed);
    
    w.resize(data[0].size());
    
    for(unsigned int t = 1; t < epochs; t++) {

        unsigned int idx = rand() % data.size();

        double nt = 1/(c*t);

        thrust::device_vector<double> xi = data[idx];

        thrust::device_vector<double> temp(xi.size(), 0);

        thrust::transform(xi.begin(), xi.end(), w.begin(), temp.begin(), thrust::multiplies<double>());

        double dot_product = thrust::reduce(temp.begin(), temp.end(), 0, thrust::plus<double>());

        thrust::device_vector<double> next_w(xi.size(), 0);

        if(dot_product*label[idx] < 1) {
            for(unsigned int k = 0; k < xi.size(); k++) {
                next_w[k] = w[k] - nt*c*w[k] + nt*label[idx]*xi[k];
            }
        }

        else {
            for(unsigned int k = 0; k < xi.size(); k++) {
                next_w[k] = w[k] - nt*c*w[k];
            }
        }

        w = next_w;
    }

    /*for(unsigned int i = 0; i < w.size(); i++) {
        cout << "w" << i << " = " << w[i] << " ";
    }*/

    cout << endl;
}

thrust::device_vector<int> SVMClassifier::predict(thrust::device_vector<thrust::device_vector<double>> & data) {
    thrust::device_vector<int> predicted_labels;

    for(unsigned int i = 0; i < data.size(); i++) {
        
        thrust::device_vector<double> xi = data[i];

        thrust::device_vector<double> temp(xi.size(), 0);
        
        thrust::transform(xi.begin(), xi.end(), w.begin(), temp.begin(), thrust::multiplies<double>());

        double dot_product = thrust::reduce(temp.begin(), temp.end(), 0, thrust::plus<double>());

        if (dot_product >= 0) {
            predicted_labels.push_back(1);
        }

        else predicted_labels.push_back(-1);
    }

    return predicted_labels;
}

double SVMClassifier::accuracy(thrust::device_vector<int> & label, thrust::device_vector<int> & pred_label) {
    int correct_pred = 0;

    
    for(unsigned int i = 0; i < label.size(); i++) {
        if (label[i] == pred_label[i]) {
            correct_pred += 1;
            //cout<< "acertou" <<endl;
        }
        //else cout<< "erou" <<endl;
    }

    return (double) correct_pred/label.size();
}