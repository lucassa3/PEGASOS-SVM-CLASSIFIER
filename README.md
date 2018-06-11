# PEGASOS-SVM-CLASSIFIER
Implementation of a support vector machine classifier using primal estimated sub-gradient solver in C++

## About
This project was about building a Suport Vector Machine binary classifier from scratch using the methods described in [this article](https://www.cs.huji.ac.il/~shais/papers/ShalevSiSrCo10.pdf), and tweaking it to a mini batch version that, in theory, would benefit from the massive parallelizing available for GPU devices.

Two main SVM dinary classifiers were then built for comparison purposes, a sequential based pegasos basic algorithm described in section 2.1 of the article, and a mini-batch version described in section 2.3. The sequential version trains with data on CPU, while the mini-batch version trains with data natively in GPU using CUDA kernels. There are also two other experimental versions using CUDA Thrust on ohter_versions folder, though be aware that they are not totally completed/optimized, and therefore wont be covered in the scope of this document.

If you would like to know more about Support vector machines and how they work, i suggest looking into this very didatic video from MIT OpenCourseWare [here](https://www.youtube.com/watch?v=_PwhiWxHK8o&t=1324s). If you want details into building the necessary environment to run this code and how to operate it, continue reading this guide, or if you are more interested on data analysis and the results i've got, skip to the Results section.

## Requirements
* have a NVIDIA GPU (this is a must, if you want to use the mini-batch version);
* g++ compiler;
* Install Boost library;
```
$ sudo apt-get install libboost-all-dev
```
* Install CUDA;
* Have nvcc compilation tool that supports your CUDA version;
* If you dont want to use the provided datasets on this repo, download it by yourself and make sure its in csv file format (doesnt doesnt have to be in .csv extension, just its comma separated value data format). Also, make sure that your sample class is represented by a number and its located in the last value on the line of the respective sample;

## How to use:
Inside each version folder there's a makefile to compile all the filed needed. If you are all set with the requirements, typing the below command shouldn't present any problems.
```
$ make
```
Once compiled, the program is ready to be runned (./main). Before you do it though, if you want to use any of the datasets provided, just unzip the dataset folder to the same place the zipped version was. If you do this and dont set any of the environment variables presented below, it will run the diabetes.csv dataset with the default parameter values (also presented below). If you dont unzip the dataset, and dont use the data_path variable, you'll get yourself a seg fault error! :(

You can use the following environment variables when running the code:
* DATA_PATH - Path to your dataset (Default: ../datasets/diabetes.csv);
* C - Lambda regularization parameter to contrrol step size (Default: 0.001);
* SAMPLES_LIMIT - If you dont want to use your whole dataset for speed purposes, specify the maximum samples you'd like to read (Default: 9999999);
* EPOCHS - Number of epochs the training part will run;
* BATCH_SIZE - The desired size of your batch (ONLY AVAILABLE IN MINI BATCH ALGORITHM) (Default: 10);
* TRAIN_SIZE - Percentage of the dataset allocated for training (0-1) (Default: 0.8);
* NUM_ITERATIONS - Number of times the algorithm will run, good to mitigate some weird results due to bad seed, and return mean accuracy of all iterations (Default: 10).

Example:
```
$  DATA_PATH=../datasets/mnist_train.csv C=0.0001 EPOCHS=500000 BATCH_SIZE=200 TRAIN_SIZE=0.8 NUM_ITERATIONS=20 ./main
```

## Results:

### 1. Performance:
The premise of building a mini-batch gpu version was that it would have a better performance as it would be able to fetch more samples at once while also benefitting from datasets that have a large number of features (e.g, MNIST). As numbers of samples/time go, this has proven to be the case:

The four graphics below represent time per number of samples of each database i used:

![Alt text](imgs/charts.png?raw=true "Title")


On the CPU sequential version (marked in red), since there is not the batch concept, the number of samples directly reflects the number of epochs the alogrithm used, limited to one single sample used per epoch. On the GPU mini-batch version (marked in blue, the number of samples are calculated using always a fixed epoch number (10000) multiplied by the number of samples contained in a batch, and increascing only the batch number at each data point.

As the charts represent, training with a small number of used samples yields a better performance on the CPU side, since the data doesnt have to be passed on to the gpu. However, as the number of samples used increase, the gpu starts compensating its slow data copy with the ability to iterate each weight at the same time, and process the whole batch in parallel as well. MNIST is clearly one that takes advantage of Mini-batch processing, given its large feature set (one per pixel in a 28x28 image). Other datasets such as Diabetes didn't get to surpass CPU version with GPU version, mainly due to the lack of samples in the dataset (only 546 in the training set), making it unable to have batches bigger than 500 samples.












