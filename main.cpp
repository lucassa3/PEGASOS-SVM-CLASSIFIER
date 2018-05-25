#include <random>
#include <vector>
#include <iostream>
#include <string>
#include <time.h>
#include <algorithm>
#include "svm_classifier.hpp"
#include "data_reader.hpp"
#include "../mnist-master/include/mnist/mnist_reader.hpp"

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

    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
    mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("../mnist-master");

    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

    vector<vector<double>> data = read_data("../iris.data.txt", 5000000,  0);

    cout << "OOOOOOO" << dataset.training_images[20][0] << endl;

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

    for (unsigned int i = 0; i < 20; i++) {

        SVMClassifier* svm_clf = new SVMClassifier(0.001, 10000000, time(NULL));
        cout << "seed: " << time(NULL) << endl;

        svm_clf->fit(x_train, y_train);

        vector<int> y_pred = svm_clf->predict(x_test);

        double cur_acc = svm_clf->accuracy(y_test, y_pred);

        //for(unsigned int j = 0; j < y_pred.size(); j++) cout << y_pred[j] << endl;

        total_acc += cur_acc;

        cout << "accurarcy: "<< cur_acc  << endl;
    }

    cout << "mean accuracy: "<< total_acc/20  << endl;

    return 0;
}