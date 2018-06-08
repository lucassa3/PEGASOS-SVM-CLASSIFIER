#include <random>
#include <vector>
#include <iostream>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/fill.h>

#include "svm_classifier.hpp"

using namespace std;

SVMClassifier::SVMClassifier(double c, unsigned int epochs, unsigned int seed, unsigned int feature_size, unsigned int batch_size) {
  this->c = c;
  this->epochs = epochs;
  this->seed = seed;
  this->feature_size = feature_size;
  this->batch_size = batch_size;
}


void SVMClassifier::fit(thrust::device_vector<double> & data, thrust::device_vector<int> & label) {
    srand(seed);  
    
    w.resize(feature_size, 0);

    thrust::device_vector<double> cur_yi_xi;
    thrust::device_vector<double> yi_xi;
    thrust::device_vector<double> xi_at;

    thrust::device_vector<double> temp(feature_size, 0);
    thrust::device_vector<double> nt_c_w(feature_size, 0);
    thrust::device_vector<double> nt_batch_yixi(feature_size, 0);
    thrust::device_vector<double> xi(feature_size*batch_size, 0); 

    //int percentage = 0;

    for(unsigned int t = 1; t < epochs; t++) {
        cur_yi_xi.clear();
        yi_xi.clear();
        xi_at.clear();

        double nt = 1/(c*t);
        double nt_batch = nt/batch_size;    

        thrust::device_vector<double> next_w(feature_size, 0);
        
        unsigned int idx = rand() % (label.size() - batch_size);
 
        //copy sample features of data[idx] to xi
        thrust::copy(data.begin()+(idx*feature_size), data.begin()+(idx*feature_size)+(batch_size*feature_size), xi.begin());
        
        // for(unsigned int i = 0; i < xi.size(); i++) {
        //     cout << "xi["<<i<<"] = " << xi[i] << endl;
        // }

        //for each sample on my batch
        for (unsigned int i = 0; i < batch_size; i++) {
            
            //get the Ith sample
            thrust::transform(xi.begin()+(i*feature_size), xi.begin()+(i*feature_size)+(feature_size), w.begin(), temp.begin(), thrust::multiplies<double>());
            double dot_product = thrust::reduce(temp.begin(), temp.end(), 0.0, thrust::plus<double>());
            
            cout << "DOT_PROD: " << dot_product*label[idx+i] << endl;
            int cur_label = label[idx+i];
            
            //if sample is positive (hence, incorrectly predicted), insert in xi_at subset to iterate over w
            if (cur_label*dot_product < 1) {
                xi_at.insert(xi_at.end(), xi.begin()+(i*feature_size), xi.begin()+(i*feature_size)+(feature_size));
            }
        }
        
        //nt*c*W
        thrust::fill(nt_c_w.begin(), nt_c_w.end(), nt*c);
        thrust::transform(w.begin(), w.end(), nt_c_w.begin(), nt_c_w.begin(), thrust::multiplies<double>());
        
        for (unsigned int i = 0; i < feature_size; i++) {
            for (unsigned int j = 0; j < xi_at.size()/feature_size; j++) {
                
                cur_yi_xi.push_back(xi_at[i + j*feature_size]*label[idx+j]);
            }

            if(xi_at.size()/feature_size != 0) {
                double yi_xi_sum = thrust::reduce(cur_yi_xi.begin(), cur_yi_xi.end(), 0.0, thrust::plus<double>());
                yi_xi.push_back(yi_xi_sum);
            }
        }

        if(yi_xi.size() == 0) {
            yi_xi.resize(feature_size, 0);
        }
        
        
        thrust::transform(w.begin(), w.end(), next_w.begin(), next_w.begin(), thrust::plus<double>());

        thrust::transform(next_w.begin(), next_w.end(), nt_c_w.begin(), next_w.begin(), thrust::minus<double>());

        thrust::fill(nt_batch_yixi.begin(), nt_batch_yixi.end(), nt_batch);
        thrust::transform(nt_batch_yixi.begin(), nt_batch_yixi.end(), yi_xi.begin(), nt_batch_yixi.begin(), thrust::multiplies<double>());

        thrust::transform(nt_batch_yixi.begin(), nt_batch_yixi.end(), next_w.begin(), next_w.begin(), thrust::plus<double>());

        thrust::copy(next_w.begin(), next_w.end(), w.begin());
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