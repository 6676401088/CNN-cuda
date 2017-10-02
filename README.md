# CNN-cuda
cuda implementation of [Convolutional Neural Networks](https://github.com/paperrune/Neural-Networks/tree/master/Convolutional_Neural_Networks)</br></br>

## Features
- Multi-GPU is not supported.
- Support Batch Normalization, Dropout and Shortcut connections for residual learning.
- To classify MNIST handwritten digits, following files are required from http://yann.lecun.com/exdb/mnist/
  - train-images.idx3-ubyte
  - train-labels.idx1-ubyte
  - t10k-images.idx3-ubyte
  - t10k-labels.idx1-ubyte
  
- To classify CIFAR-10 datasets, following files are required from https://www.cs.toronto.edu/~kriz/cifar.html
  - data_batch_1.bin
  - data_batch_2.bin
  - data_batch_3.bin
  - data_batch_4.bin
  - data_batch_5.bin
  - test_batch.bin

- The network structure is determined by three variables in the main.cpp.

  ```C++
  137: char *type_layer[] = {"CIFAR-10", "Cbn,fs3 /sc",
                              "Cbn,fs3",     "Cbn,fs3 /sc2",      "Cbn,fs3", "Cbn,fs3 /sc2", "Cbn,fs3", "Cbn,fs3 /sc2",
                              "Cbn,fs3,st2", "Cbn,fs3 /psc2,st2", "Cbn,fs3", "Cbn,fs3 /sc2", "Cbn,fs3", "Cbn,fs3 /sc2",
                              "Cbn,fs3,st2", "Cbn,fs3 /psc2,st2", "Cbn,fs3", "Cbn,fs3 /sc2", "Cbn,fs3", "Cbn,fs3 /sc2",
                              "Pavg", "Lce,sm"};
  144: int length_map[]    = {32,	32, 32, 32, 32, 32, 32, 32, 16, 16, 16, 16, 16, 16,  8,  8,  8,  8,  8,  8,  1,  1};
  145: int number_map[]    = { 3, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64, 10};
  ```  
  - There is no type for input layer but "MNIST" and "CIFAR-10" are used to select the data to be read in the main
  - Type start with 'C(connecting/convolution)' and 'P(padding/pooling)' is for hidden layer.
  
  	```
    C(connecting/convolution)
    > Activation Function
    "ls" : Logistic Sigmoid
    "ht" : Hyperbolic Tangent
    ""   : default is ReLU
    
    > Shourtcut connections (original version https://arxiv.org/pdf/1512.03385.pdf)
    "/sc"       : start shortcut connections
    "/scn"      : identity shortcut connected to the (i - n)th layer
    "/pscn"     : projection shortcut connected to the (i - n)th layer for increasing dimension or reducing feature map size
    "/pscn,stn" : stride can be set in case of projection shortcut
    
    > Property
    "fsn"  : setting filter size to n^2  [default filter size : (length_map[i - 1] - length_map[i] + 1)^2]
    "stn"  : setting stride to n         [default stride      : 1]

    > Regularization
    "bn"   : Batch Normalization
    "do.f" : Dropout with rate 0.f, each neurons is set to zero with a probability of (1 - 0.f)
    ----------------------------------------------------------------------------------------------------
    P(padding/pooling)
    > Type
    "avg"  : Average Pooling
    "max"  : Max Pooling
    "pad"  : Zero Padding (it should be used to increase the size of the feature map)
    
    > Shourtcut connections
    "/sc"  : start shortcut connections
    
    stride and pooling size is (length_map[i - 1] / length_map[i])^2 and overlapped pooling is not supported.
	  ```
   - Type start with 'L(loss)' is for output layer.
   
	 ```
	 > Loss Function
	 "ce"  : Cross Entropy
	 "mse" : Mean Squared Error
	 
	 > Activation Function for "ce"
	 "sm"  : Softmax
	 ""    : default is Logistic Sigmoid

	 > Activation Function for "mse"
	 "ht"  : Hyperbolic Tangent
	 "ia"  : Identity Activation f(x) = x
	 ""    : default is Logistic Sigmoid
	 ```
</br>

## CIFAR-10 classification result (without data augmentation)
![result](/result.png)
