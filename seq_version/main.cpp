#include <random>
#include <vector>
#include <iostream>
#include <string>
#include <time.h>
#include <algorithm>
#include "svm_classifier.hpp"
#include "data_reader.hpp"

using namespace std;

void set_envs(string & data_path, double *c, unsigned int *samples_limit, unsigned int *epochs, unsigned int *train_size, unsigned int *num_iterations) {
    
    char *data_path_env, *samples_limit_env, *epochs_env, *train_size_env, *num_iterations_env;

    data_path_env = getenv ("DATA_PATH");
    if(data_path_env != NULL) {
        data_path = data_path_env;
    }
     
    samples_limit_env = getenv ("SAMPLES_LIMIT");
    if(samples_limit_env != NULL) {
        *samples_limit = atoi(samples_limit_env);
    }

    epochs_env = getenv ("EPOCHS");
    if(epochs_env != NULL) {
        *epochs = atoi(epochs_env);
    }

    train_size_env = getenv ("TRAIN_SIZE");
    if(train_size_env != NULL) {
        *train_size = atoi(train_size_env);
    }

    num_iterations_env = getenv ("NUM_ITERATIONS");
    if(num_iterations_env != NULL) {
        *num_iterations = atoi(num_iterations_env);
    }
}


int main(int argc, char *argv[]) {
    string data_path = "../datasets/diabetes.csv";
    double c = 0.001;
    unsigned int samples_limit = 9999999;
    unsigned int epochs = 100000;
    unsigned int train_size = 5;
    unsigned int num_iterations = 5;

    set_envs(data_path, &c, &samples_limit, &epochs, &train_size, &num_iterations);
    
    vector<vector<double>> data = read_data(data_path.c_str(), samples_limit);

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

    

    vector<vector<double>> x_test(data.begin(), data.begin() + data.size()/train_size);
    vector<vector<double>> x_train(data.begin() + data.size()/train_size, data.end());
 //
    vector<int> y_test(labels.begin(), labels.begin() + labels.size()/train_size);
    vector<int> y_train(labels.begin() + labels.size()/train_size, labels.end());

    cout << x_train.size() << endl;
    cout << y_train.size() << endl;


    double total_acc = 0;
    vector<int> y_pred;

    for (unsigned int i = 0; i < num_iterations; i++) {

        SVMClassifier* svm_clf = new SVMClassifier(c, epochs, time(NULL));
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

    cout << "mean accuracy: "<< total_acc/num_iterations  << endl;

    return 0;
}