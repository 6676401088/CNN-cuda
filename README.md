# CNN
CUDA implementation of Convolutional Neural Networks classifying MNIST and CIFAR-10 datasets.</br></br>

## Features
- Support Batch Normalization, Dropout and Shortcut connections for residual learning.
- To classify MNIST handwritten digits, followed files are required from http://yann.lecun.com/exdb/mnist/
  - train-images.idx3-ubyte
  - train-labels.idx1-ubyte
  - t10k-images.idx3-ubyte
  - t10k-labels.idx1-ubyte
  
- To classify CIFAR-10 datasets, followd files are required from https://www.cs.toronto.edu/~kriz/cifar.html<br><br>
  CIFAR-10 binary version (suitable for C programs)
  - data_batch_1.bin
  - data_batch_2.bin
  - data_batch_3.bin
  - data_batch_4.bin
  - data_batch_5.bin
  - test_batch.bin

- The network structure is determined by three parameters in the main.cpp at initialization.

  ```C++
  146: char *type_layer[] = {"CIFAR-10", "Cbn,fs3 /sc",
                              "Cbn,fs3",     "Cbn,fs3 /sc2",      "Cbn,fs3", "Cbn,fs3 /sc2", "Cbn,fs3", "Cbn,fs3 /sc2",
                              "Cbn,fs3,st2", "Cbn,fs3 /psc2,st2", "Cbn,fs3", "Cbn,fs3 /sc2", "Cbn,fs3", "Cbn,fs3 /sc2",
                              "Cbn,fs3,st2", "Cbn,fs3 /psc2,st2", "Cbn,fs3", "Cbn,fs3 /sc2", "Cbn,fs3", "Cbn,fs3 /sc2",
                              "Pavg", "Lce,sm"};
  153: int length_map[]    = {32,	32, 32, 32, 32, 32, 32, 32, 16, 16, 16, 16, 16, 16,  8,  8,  8,  8,  8,  8,  1,  1};
  154: int number_map[]    = { 3, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64, 10};
  ```  
  - There is no type for input layer but "MNIST" and "CIFAR-10" are used to select the data to be read in main()
  - Type start with 'C(connecting/convolution)' and 'P(pooling)' is for hidden layer.
  
  	```
    C(connecting/convolution)
    > Activation Function
    "ls"  : Logistic Sigmoid
    "ht"  : Hyperbolic Tangent
    ""    : default is ReLU
    
    > Shourtcut connections (original version https://arxiv.org/pdf/1512.03385.pdf)
    "/sc"   : starting layer
    "/scn"  : identity shortcut connected to the (i - n)th layer
    "/pscn" : projection shortcut connected to the (i - n)th layer for increasing dimension or reducing feature map size
    
    > Property
    "fsn" : setting filter size to n^2  [default filter size : (length_map[i - 1] - length_map[i] + 1)^2]
    "stn" : setting stride to n         [default stride      : 1]

    > Regularization
    "bn"    : Batch Normalization
    "do.f"  : Dropout with rate 0.f, each neurons is set to zero with a probability of (1 - 0.f)
    ----------------------------------------------------------------------------------------------------
    P(pooling)
    > Type
    "avg" : Average Pooling
    "max" : Max Pooling
    "pad" : Zero Padding (it should be used to increase the size of the feature map)
    
    stride and pooling size is (length_map[i - 1] / length_map[i])^2    
    overlapped pooling is not supported.
	  ```
   - Type start with 'L(loss)' is for output layer.
   
	 ```
	 > Loss Function
	 "ce"   : Cross Entropy
	 "mse"	: Mean Squared Error
	 
	 > Activation Function for "ce"
	 "sm"	: Softmax
	 ""   : default is Logistic Sigmoid

	 > Activation Function for "mse"
	 "ht"	: Hyperbolic Tangent
	 "ia"	: Identity Activation f(x) = x
	 ""   : default is Logistic Sigmoid
	 ```
</br>

## CIFAR-10 classification results
![result](/result.PNG)
