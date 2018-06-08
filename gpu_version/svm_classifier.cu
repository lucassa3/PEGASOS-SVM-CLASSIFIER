#include <random>
#include <vector>
#include <iostream>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/fill.h>

#include "svm_classifier.hpp"

SVMClassifier::SVMClassifier(double c, unsigned int epochs, unsigned int seed, unsigned int feature_size) {
  this->c = c;
  this->epochs = epochs;
  this->seed = seed;
  this->feature_size = feature_size;
}


void SVMClassifier::fit(thrust::device_vector<double> & data, thrust::device_vector<int> & label) {
    srand(seed);  
    
    w.resize(feature_size, 0);

     

    for(unsigned int t = 1; t < epochs; t++) {

        thrust::device_vector<double> temp(feature_size, 0);
        thrust::device_vector<double> nt_c_w(feature_size, 0);
        thrust::device_vector<double> nt_label_xi(feature_size, 0);
        
        unsigned int idx = rand() % label.size();
 
        int cur_label = label[idx];
        double nt = 1/(c*t);
        double nt_c = nt*c;
        double nt_label = nt*cur_label;

        thrust::device_vector<double> xi(feature_size, 0);
        //copy sample features of data[idx] to xi
        thrust::copy(data.begin()+(idx*feature_size), data.begin()+(idx*feature_size)+(feature_size), xi.begin());
 
        // for(unsigned int i = 0; i < xi.size(); i++) {
        //     cout << "xi["<<i<<"] = " << xi[i] << endl;
        // }

        thrust::device_vector<double> next_w(feature_size, 0);

        //dot_prduct W*Xi
        thrust::transform(xi.begin(), xi.end(), w.begin(), temp.begin(), thrust::multiplies<double>());
        double dot_product = thrust::reduce(temp.begin(), temp.end(), 0, thrust::plus<double>());
        
        //cout << "DOT_PROD: " << dot_product <<endl;
        //nt*c*W
        thrust::fill(nt_c_w.begin(), nt_c_w.end(), nt_c);
        thrust::transform(w.begin(), w.end(), nt_c_w.begin(), nt_c_w.begin(), thrust::multiplies<double>());

        //nt*cur_label*XI
        thrust::fill(nt_label_xi.begin(), nt_label_xi.end(), nt_label);
        thrust::transform(xi.begin(), xi.end(), nt_label_xi.begin(), nt_label_xi.begin(), thrust::multiplies<double>());


        if(dot_product*cur_label < 1) {
            //next_w = + nt*label[idx]*xi[k] - nt*c*w[k] + w[k]   
            thrust::transform(nt_label_xi.begin(), nt_label_xi.end(), next_w.begin(), next_w.begin(), thrust::plus<double>());
            thrust::transform(next_w.begin(), next_w.end(), nt_c_w.begin(), next_w.begin(), thrust::minus<double>());
            thrust::transform(w.begin(), w.end(), next_w.begin(), next_w.begin(), thrust::plus<double>());
        }

        else {
            //next_w = - nt*c*w[k] + w[k]
            thrust::transform(next_w.begin(), next_w.end(), nt_c_w.begin(), next_w.begin(), thrust::minus<double>());
            thrust::transform(w.begin(), w.end(), next_w.begin(), next_w.begin(), thrust::plus<double>());
        }

        thrust::copy(next_w.begin(), next_w.end(), w.begin());






        // for(unsigned int i = 0; i < w.size(); i++) {
        //     cout << "w" << i << " = " << w[i] << endl;
        // }

    }

    for(unsigned int i = 0; i < w.size(); i++) {
        cout << "w" << i << " = " << w[i] << endl;
    }

    cout << endl;
}

thrust::device_vector<int> SVMClassifier::predict(thrust::device_vector<double> & data) {
    thrust::device_vector<int> predicted_labels;

    for(unsigned int i = 0; i < data.size()/feature_size; i++) {

        thrust::device_vector<double> xi(data.begin()+(i*feature_size), data.begin()+(i*feature_size)+(feature_size));
        
        thrust::device_vector<double> temp(feature_size, 0);
        
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
        }
    }

    return (double) correct_pred/label.size();
}