#include <random>
#include <vector>
#include <iostream>
#include <string>
#include <time.h>
#include "svm_classifier.hpp"

SVMClassifier::SVMClassifier(double c, unsigned int epochs, unsigned int seed) {
  this->c = c;
  this->epochs = epochs;
  this->seed = seed;
}


void SVMClassifier::fit(vector<vector<double>> & data, vector<int> & label) {
    srand(seed);
    
    w.resize(data[0].size());
    
    for(unsigned int t = 1; t < epochs; t++) {

        unsigned int idx = rand() % data.size();

        double nt = 1/(c*t);

        vector<double> xi = data[idx];

        vector<double> next_w(xi.size(), 0);

        double dot_product = 0;
        

        for(unsigned int i = 0; i < xi.size(); i++) {   
            dot_product += w[i]*xi[i];
        }

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

    // for(unsigned int i = 0; i < w.size(); i++) {
    //     cout << "w" << i << " = " << w[i] << endl;
    // }

    cout << endl;
}

vector<int> SVMClassifier::predict(vector<vector<double>> & data) {
    vector<int> predicted_labels;

    for(unsigned int i = 0; i < data.size(); i++) {
        
        vector<double> xi = data[i];
        
        double dot_product = 0;
       
        for(unsigned int j = 0; j < xi.size(); j++) {   
            dot_product += w[j]*xi[j];
        }

        if (dot_product >= 0) {
            predicted_labels.push_back(1);
        }

        else predicted_labels.push_back(-1);
    }

    return predicted_labels;
}

double SVMClassifier::accuracy(vector<int> & label, vector<int> & pred_label) {
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