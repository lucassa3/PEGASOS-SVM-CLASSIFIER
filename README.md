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







