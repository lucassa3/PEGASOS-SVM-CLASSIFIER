#ifndef SVM_CLASS_H
#define SVM_CLASS_H

#include <vector>
#include <thrust/device_vector.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

class SVMClassifier {

  double * w;
  double c;
  unsigned int epochs;
  unsigned int seed;
  unsigned int feature_size;
  unsigned int batch_size;

public:

  SVMClassifier(double c, unsigned int epochs, unsigned int seed, unsigned int feature_size, unsigned int batch_size);

  void fit(thrust::device_vector<double> & data, thrust::device_vector<int> & label);

  thrust::device_vector<int> predict(thrust::device_vector<double> & data);

  double accuracy(thrust::device_vector<int> & label, thrust::device_vector<int> & pred_label);

};

template <typename T>
struct KernelArray {
  T*  _array;
  int _size;
};

template <typename T>
KernelArray<T> convertToKernel(thrust::device_vector<T> & dVec) {
  
  KernelArray<T> kArray;
  kArray._array = thrust::raw_pointer_cast(&dVec[0]);
  kArray._size  = (int) dVec.size();

  return kArray;
}

template <typename T>
KernelArray<T> convertToKernel2(thrust::device_vector<T> & dVec) {
  
  KernelArray<T> kArray;
  kArray._array = thrust::raw_pointer_cast(&dVec[0]);
  kArray._size  = (int) dVec.size();

  return kArray;
}

#endif