#include <random>
#include <vector>
#include <iostream>
#include <string>
#include <time.h>
#include <algorithm>
#include "svm_classifier.hpp"
#include "data_reader.hpp"

using namespace std;

double rand_float(double low, double high) {
    
    return ((double)rand() * (high - low)) / (double)RAND_MAX + low;
} 

void generate_random_dataset(vector<vector<double>> & data, vector<int> & label) {  
    
    for (unsigned int i = 0; i < data.size(); i++) {
        
        vector<double> aux_vec;

        for (unsigned int j = 0; j < data[0].size(); j++) {
            
            if (j == data[0].size() -1) {
                aux_vec.push_back(1);
            }

            else {
                if (i%2) {
                    aux_vec.push_back(rand_float(0,30));
                }

                else {
                    aux_vec.push_back(rand_float(10,40));
                }
            }    
        }

        data.push_back(aux_vec);

        if (i%2) {
            label.push_back(1);
        }

        else {
            label.push_back(-1);
        }
    }
}


int main(int argc, char *argv[]) {
    
    vector<vector<double>> data = read_data("../datasets/iris.data.txt", 5000000,  0);

    // for(unsigned int i = 0; i < dataset.training_images.size(); i++) {
    //     for(unsigned int j = 0; j < dataset.training_images[i].size(); j++) cout << dataset.training_images[i][j] << " ";
    //     cout << endl;
    // }
    
    random_shuffle(data.begin(), data.end());
    /*for(unsigned int i = 0; i < data.size(); i++) {
        for(unsigned int j = 0; j < data[i].size(); j++) cout << data[i][j] << " ";
        cout << endl;
    }*/
    
    vector<double> labels = set_labels(data);

    //for(unsigned int j = 0; j < labels.size(); j++) cout << labels[j] << endl;

    

    vector<vector<double>> x_train(data.begin(), data.begin() + data.size()/2);
    vector<vector<double>> x_test(data.begin() + data.size()/2, data.end());

    vector<int> y_train(labels.begin(), labels.begin() + labels.size()/2);
    vector<int> y_test(labels.begin() + labels.size()/2, labels.end());


    double total_acc = 0;
    vector<int> y_pred;

    for (unsigned int i = 0; i < 1; i++) {

        SVMClassifier* svm_clf = new SVMClassifier(0.01, 20000, 42);
        //cout << "seed: " << time(NULL) << endl;

        svm_clf->fit(x_train, y_train);

        y_pred = svm_clf->predict(x_test);

        double cur_acc = svm_clf->accuracy(y_test, y_pred);

        //

        total_acc += cur_acc;

        cout << "accuracy: "<< cur_acc  << endl;
    }

    // for(unsigned int j = 0; j < y_pred.size(); j++) {
    //     cout << y_pred[j];
    //     cout << " ";
    //     cout<< y_test[j] << endl;

    // }

    cout << "mean accuracy: "<< total_acc/1  << endl;

    return 0;
}