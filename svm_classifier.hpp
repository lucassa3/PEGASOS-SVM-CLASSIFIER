#ifndef SVM_CLASS_H
#define SVM_CLASS_H

#include <vector>
#include <time.h>

using namespace std;

class SVMClassifier {

  vector<double> w;
  double c;
  unsigned int epochs;
  unsigned int seed;

public:

  SVMClassifier(double c, unsigned int epochs, unsigned int seed);

  void fit(vector<vector<double>> & data, vector<int> & label);

  vector<int> predict(vector<vector<double>> & data);

  double  accuracy(vector<int> & label, vector<int> & pred_label);

};

#endif