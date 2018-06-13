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

__global__ void predict_array(KernelArray<double> kArray, double *w, int * d_y_pred, unsigned int feature_size) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    
    double dot_product = 0;
    
    for(int j = 0; j < feature_size; j++) {
        dot_product += w[j]*kArray._array[i*feature_size + j];
    }

    if (dot_product >= 0) {
        d_y_pred[i] = 1;
    }

    else d_y_pred[i] = -1;
}

__global__ void copy_batch(KernelArray<double> kArray, double *xi, unsigned int idx, unsigned int feature_size, unsigned int batch_size) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i < feature_size*batch_size) {
        xi[i] = kArray._array[idx*feature_size+i];
    }
}

__global__ void copy_array(double*a, double *b, unsigned int feature_size) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i < feature_size) {
        a[i] = b[i];
    }
    
}

__global__ void select_samples(double *xi, double *w, unsigned int feature_size, unsigned int idx, KernelArray<int> label, unsigned int batch_size) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i < batch_size) {
        double dot_product = 0;
    
        for(int j = 0; j < feature_size; j++) {
            dot_product += w[j]*xi[i*feature_size + j];
        }

        if(dot_product*label._array[idx+i] >= 1) {
            for(int j = 0; j < feature_size; j++) {
                xi[i*feature_size + j] = 0;
            }
        }
    }
    
}

__global__ void reduce_by_samples(double *yi_xi, double *xi, unsigned int batch_size, unsigned int feature_size, unsigned int idx, KernelArray<int> label) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;

    if(i < feature_size) {
        double sum = 0;
    
        for(int j = 0; j < batch_size; j++) {
            sum += xi[j*feature_size + i]*label._array[idx+j]; //acho que esta OK
        }

        yi_xi[i] = sum;
    }
    
}

__global__ void update_w(double *yi_xi, double *w, double *next_w, double nt, double c, int batch_size, unsigned int feature_size) {

    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i < feature_size) {
        next_w[i] = w[i] - nt*c*w[i] + (nt/batch_size)*yi_xi[i];
    } 
}

void SVMClassifier::fit(thrust::device_vector<double> & data, thrust::device_vector<int> & label) {
    srand(seed);  
    
    cudaMalloc((void **)&w, feature_size*sizeof(double));
    cudaMemset(w, 0, feature_size*sizeof(double));

    double * xi;
    cudaMalloc((void **)&xi, batch_size*feature_size*sizeof(double));

    double * yi_xi;
    cudaMalloc((void **)&yi_xi, feature_size*sizeof(double));

    double * next_w;
    cudaMalloc((void **)&next_w, feature_size*sizeof(double));

    //convert device_vector to gpu array
    KernelArray<double> ka_data = convertToKernel(data);
    KernelArray<int> ka_label = convertToKernel(label);

    for(unsigned int t = 1; t < epochs; t++) {
        cudaMemset(next_w, 0, feature_size*sizeof(double));

        double nt = 1/(c*t);    
        unsigned int idx = rand() % (label.size() - batch_size);


        copy_batch<<<ceil(batch_size*feature_size/512.0), 512>>>(ka_data, xi, idx, feature_size, batch_size);
        

        select_samples<<<ceil(batch_size/512.0), 512>>>(xi, w, feature_size, idx, ka_label, batch_size);


        reduce_by_samples<<<ceil(feature_size/512.0), 512>>>(yi_xi, xi, batch_size, feature_size, idx, ka_label);


        update_w<<<ceil(feature_size/512.0), 512>>>(yi_xi, w, next_w, nt, c, batch_size, feature_size);


        copy_array<<<ceil(feature_size/512.0), 512>>>(w, next_w, feature_size);

    }

    // double *h_b;
    // h_b = (double *)malloc(feature_size*sizeof(double));
    // cudaMemcpy(h_b, w, feature_size*sizeof(double), cudaMemcpyDeviceToHost);
    // for(unsigned int i = 0; i < feature_size; i++) {
    //     cout << "w["<<i<<"] = " << h_b[i] << endl;
    // }

    cudaFree(xi);
    cudaFree(yi_xi);
    cudaFree(next_w);

    cout << endl;
}

thrust::device_vector<int> SVMClassifier::predict(thrust::device_vector<double> & data) {
    thrust::device_vector<int> predicted_labels;
    
    int *d_predicted_labels;
    cudaMalloc((void **)&d_predicted_labels, data.size()/feature_size*sizeof(int));


    KernelArray<double> ka_data = convertToKernel(data);

    predict_array<<<ceil((data.size()/feature_size)/512.0), 512>>>(ka_data, w, d_predicted_labels, feature_size);

    int *y_pred;
    y_pred = (int *)malloc(data.size()/feature_size*sizeof(int));
    cudaMemcpy(y_pred, d_predicted_labels, data.size()/feature_size*sizeof(int), cudaMemcpyDeviceToHost);

    for(unsigned int i = 0; i < data.size()/feature_size; i++) {
        predicted_labels.push_back(y_pred[i]);
    }

    cudaFree(w);
    cudaFree(d_predicted_labels);
    free(y_pred);


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

