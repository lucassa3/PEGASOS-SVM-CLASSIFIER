#ifndef SVM_CLASS_H
#define SVM_CLASS_H

#include <vector>
#include <thrust/device_vector.h>

using namespace std;

class SVMClassifier {

  thrust::device_vector<double> w;
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

#endif