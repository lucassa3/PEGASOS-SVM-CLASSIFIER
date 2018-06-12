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

void set_envs(string & data_path, double *c, unsigned int *samples_limit, unsigned int *epochs, unsigned int *batch_size, unsigned int *train_size, unsigned int *num_iterations) {
    
    char *data_path_env, *c_env, *samples_limit_env, *epochs_env, *batch_size_env, *train_size_env, *num_iterations_env;

    data_path_env = getenv ("DATA_PATH");
    if(data_path_env != NULL) {
        data_path = data_path_env;
    }

    c_env = getenv ("C");
    if(c_env != NULL) {
        *c = atof(c_env);
    }
     
    samples_limit_env = getenv ("SAMPLES_LIMIT");
    if(samples_limit_env != NULL) {
        *samples_limit = atoi(samples_limit_env);
    }

    epochs_env = getenv ("EPOCHS");
    if(epochs_env != NULL) {
        *epochs = atoi(epochs_env);
    }

    batch_size_env = getenv ("BATCH_SIZE");
    if(batch_size_env != NULL) {
        *batch_size = atoi(batch_size_env);
    }

    train_size_env = getenv ("TRAIN_SIZE");
    if(train_size_env != NULL) {
        *train_size = 1/(1-atof(train_size_env));
    }

    num_iterations_env = getenv ("NUM_ITERATIONS");
    if(num_iterations_env != NULL) {
        *num_iterations = atoi(num_iterations_env);
    }
}

int main(int argc, char *argv[]) {
    cudaEvent_t start, stop, iter_start, iter_stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&iter_start);
    cudaEventCreate(&iter_stop);

    string data_path = "../datasets/diabetes.csv";
    double c = 0.001;
    unsigned int samples_limit = 9999999;
    unsigned int epochs = 100000;
    unsigned int batch_size = 10;
    unsigned int train_size = 5;
    unsigned int num_iterations = 5;

    set_envs(data_path, &c, &samples_limit, &epochs, &batch_size, &train_size, &num_iterations);

    cout << "Reading and parsing data: " << endl;

    vector<vector<double>> h_data = read_data(data_path.c_str(), samples_limit);
    
    const unsigned int f_size = feature_size(h_data);
    
    random_shuffle(h_data.begin(), h_data.end());

    vector<double> h_labels = set_labels(h_data);

    cout << "Done " << endl;

    cout << "Sending data to gpu: " << endl;
    
    thrust::device_vector<double> data;

    for (vector<double> sample: h_data) {
        thrust::device_vector<double> aux(sample);
        data.insert(data.end(), aux.begin(), aux.end());   
    }

    thrust::device_vector<double> labels(h_labels);

    thrust::device_vector<double> x_test(data.begin(), data.begin() + (data.size()/train_size) - ((data.size()/train_size)%f_size));
    thrust::device_vector<double> x_train(data.begin() + (data.size()/train_size) - ((data.size()/train_size)%f_size), data.end());

    thrust::device_vector<int> y_test(labels.begin(), labels.begin() + labels.size()/train_size);
    thrust::device_vector<int> y_train(labels.begin() + labels.size()/train_size, labels.end());

    cout << "Done " << endl;
    
    // cout << x_train.size() << endl;
    // cout << y_train.size() << endl;

    double total_acc = 0;

    thrust::device_vector<int> y_pred(y_test.size());
    
    cudaEventRecord(start, NULL);

    cout << "Started SVM: " << endl;

    for (unsigned int i = 0; i < num_iterations; i++) {

        

        SVMClassifier* svm_clf = new SVMClassifier(c, epochs, time(NULL)+i, f_size, batch_size);
        //cout << "seed: " << time(NULL) << endl;

        cudaEventRecord(iter_start, NULL);

        // cout << "Fitting data: " << endl;
        svm_clf->fit(x_train, y_train);
        // cout << "Done " << endl;

        cudaEventRecord(iter_stop, NULL);
        cudaEventSynchronize(iter_stop);
        float iter_msecTotal = 0.0f;
        cudaEventElapsedTime(&iter_msecTotal, iter_start, iter_stop);
        cout << "Current iteration training time: " << iter_msecTotal/1000 << " seconds" << endl;

        // cout << "Predicting data: " << endl;
        y_pred = svm_clf->predict(x_test);
        // cout << "Done " << endl;

        double cur_acc = svm_clf->accuracy(y_test, y_pred);

        total_acc += cur_acc;

        cout << "accuracy: "<< cur_acc  << endl;
        
        

       

    }

    cout << endl << "mean accuracy: "<< total_acc/num_iterations  << endl;

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    cout << "Elapsed total training + predicting time: " << msecTotal/1000 << " seconds" << endl;

    return 0;
}