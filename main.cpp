#include <random>
#include <vector>
#include <iostream>
#include <string>
#include <time.h>
#include "svm_classifier.hpp"

#define DATASET_LENGTH 2000
#define FEATURE_NUM 6

using namespace std;

double rand_float( double low, double high ) {
    return ( ( double )rand() * ( high - low ) ) / ( double )RAND_MAX + low;
}

void generate_random_dataset(vector<vector<double>> & data, vector<int> & label) {
    
    for (unsigned int i = 0; i < DATASET_LENGTH; i++) {
        vector<double> aux_vec;

        for (unsigned int j = 0; j < FEATURE_NUM; j++) {
            if (j == FEATURE_NUM -1) {
                aux_vec.push_back(1);
            }

            else {
                if (i%2) {
                    aux_vec.push_back(rand_float(0,100));
                }

                else {
                    aux_vec.push_back(rand_float(100,200));
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


    vector<vector<double>> feature_set;
    vector<int> label_set;

    generate_random_dataset(feature_set, label_set);

    vector<vector<double>> feature_set2;
    vector<int> label_set2;

    for (unsigned int i = 0; i < DATASET_LENGTH; i++) {
        vector<double> aux_vec;

        for (unsigned int j = 0; j < FEATURE_NUM; j++) {
            if (j == FEATURE_NUM -1) {
                aux_vec.push_back(1);
            }

            else {
                if (i%2) {
                    aux_vec.push_back(rand_float(0,105));
                }

                else {
                    aux_vec.push_back(rand_float(95,200));
                }
            }    
        }

        feature_set2.push_back(aux_vec);
        if (i%2) {
            label_set2.push_back(1);
        }

        else {
            label_set2.push_back(-1);
        }
    }

    for (unsigned int i = 0; i < DATASET_LENGTH; i++) {
        vector<double> aux_vec;

        for (unsigned int j = 0; j < FEATURE_NUM; j++) {
                cout << feature_set[i][j] << " ";
        }

        cout << endl;

        cout << "label: " << label_set[i] << endl;
    }

    SVMClassifier* svm_clf = new SVMClassifier(0.01, 100000, time(NULL));

    svm_clf->fit(feature_set, label_set);
    vector<int> oi = svm_clf->predict(feature_set2);
    cout << svm_clf->accuracy(label_set2, oi) << endl;
    
    return 0;
}