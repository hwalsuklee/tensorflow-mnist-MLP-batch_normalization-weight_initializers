# Various initializers and batch normalization

An implementation of various initializers for weight and bias and batch normalization in Tensorflow.

MNIST database is used to show performance-comparison

## Network architecture

In order to examine the effect of initializers and batch normalization, a simple network architecture called multi-layer perceptrons (MLP) is employed.

MLP has following architecture.

+ input layer : 784 nodes (MNIST images size)
+ first hidden layer : 256 nodes
+ second hidden layer : 256 nodes
+ output layer : 10 nodes (number of class for MNIST)

## Various initializers

The following initializers for weights and biases of network are considered.

+ Weight Initializer
  * normal, trucated normal
  * xaiver : [Understanding the difficulty of training deep feedforward neural networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
  * he : [Delving Deep into Rectifiers](http://arxiv.org/pdf/1502.01852v1.pdf)
+ Bias Initializer
  * normal, constant
  
### Simulation results
+ Weight Initializer : he > trauncated normal = xaiver > normal
+ Bias Initilaizer : zero > normal

Sample results are following.

|Index|Weight Initializer|Bias Initializer|Accuracy|
|:---:|:---:|:---:|:---:|
|normal_w_normal_b_0.9451|normal|normal|0.9451|
|normal_w_zero_b_0.9485|normal|zero|0.9485|
|truncated_normal_w_normal_b_0.9788|truncated_normal|normal|0.9788|
|truncated_normal_w_zero_b_0.9790|truncated_normal|zero|0.9790|
|xavier_w_normal_b_0.9800|xavier|normal|0.9800|
|xavier_w_zero_b_0.9806|xavier|zero|0.9806|
|he_w_normal_b_0.9798|he|normal|0.9798|
|he_w_zero_b_0.9811|he|zero|0.9811|

## Batch normalization
Batch normalization improves performance of network in terms of final accuracy and convergence rate.</br>
In this simulation, bias initalizer was zero-constant initializer and weight initializers were xavier / he.

Sample results are following.

|Index|Weight Initializer|Batch Normalization|Accuracy|
|:---:|:---:|:---:|:---:|
|xavier_woBN_0.9806|xavier|Unused|0.9806|
|xavier_withBN_0.9812|xavier|Used|0.9812|
|he_woBN_0.9811|he|Unused|0.9811|
|he_withBN_0.9837|he|Used|0.9837|
