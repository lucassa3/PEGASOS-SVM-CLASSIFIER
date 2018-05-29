#ifndef SVM_CLASS_H
#define SVM_CLASS_H

#include <vector>
#include <time.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>

using namespace std;

class SVMClassifier {

  thrust::device_vector<double> w;
  double c;
  unsigned int epochs;
  unsigned int seed;

public:

  SVMClassifier(double c, unsigned int epochs, unsigned int seed);

  void fit(thrust::device_vector<double> & data, thrust::device_vector<int> & label);

  thrust::device_vector<int> predict(thrust::device_vector<double> & data);

  double  accuracy(thrust::device_vector<int> & label, thrust::device_vector<int> & pred_label);

};

#endif