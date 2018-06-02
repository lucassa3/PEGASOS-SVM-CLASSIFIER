#include <random>
#include <vector>
#include <iostream>
#include <string>
#include <time.h>
#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>

#include "svm_classifier.hpp"
#include "data_reader.hpp"

using namespace std;

int main(int argc, char *argv[]) {
    
    vector<vector<double>> h_data = read_data("../datasets/diabetes.csv", 5000000,  0);
    
    const unsigned int f_size = feature_size(h_data);
    
    random_shuffle(h_data.begin(), h_data.end());

    vector<double> h_labels = set_labels(h_data);
    
    thrust::device_vector<double> data;

    for (vector<double> sample: h_data) {
        thrust::device_vector<double> aux(sample);
        data.insert(data.end(), aux.begin(), aux.end());   
    }

    thrust::device_vector<double> labels(h_labels);

    //thrust::copy(data.begin(), data.begin()+9, std::ostream_iterator<int>(std::cout, "\n"));  
    //thrust::copy(labels.begin(), labels.begin()+1, std::ostream_iterator<int>(std::cout, "\n"));

    thrust::device_vector<double> x_train(data.begin(), data.begin() + (data.size()/2) - ((data.size()/2)%9));
    thrust::device_vector<double> x_test(data.begin() + (data.size()/2) - ((data.size()/2)%9), data.end());

    thrust::device_vector<int> y_train(labels.begin(), labels.begin() + labels.size()/2);
    thrust::device_vector<int> y_test(labels.begin() + labels.size()/2, labels.end());
    
    cout << x_train.size() << endl;
    cout << y_train.size() << endl;

    double total_acc = 0;

    thrust::device_vector<int> y_pred(y_test.size());

    for (unsigned int i = 0; i < 5; i++) {

        SVMClassifier* svm_clf = new SVMClassifier(0.001, 10000, time(NULL), f_size, 80);
        //cout << "seed: " << time(NULL) << endl;

        svm_clf->fit(x_train, y_train);

        y_pred = svm_clf->predict(x_test);

        double cur_acc = svm_clf->accuracy(y_test, y_pred);

        total_acc += cur_acc;

        cout << "accurarcy: "<< cur_acc  << endl;
    }

    cout << "mean accuracy: "<< total_acc/5  << endl;

    for(unsigned int j = 0; j < y_train.size(); j++) cout << y_pred[j];

    return 0;
}