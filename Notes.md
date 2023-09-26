# Neural Networks and Deep Learning

tags: #AI #development #neuralNetworks #programming 
link: [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html)

## Chapter 1
Chapter 1 talks about the Perceptron, which is the most basic neuron. It takes binary inputs with associated weights and computes a binary output with a bias. It then extends this to Sigmoid's, these are like the Perceptron however can have any value between 0 and 1. The output is calculated with the Sigmoid function:

$$\sigma(z) = \frac{1}{1+e^{-z}}$$
Where z is calculated as the following: 
$$z=\sum_jw_jx_j-b$$
These equations give a value of close to 1 for large z and close to 0 for small, hence giving our Sigmoid neuron an output of between 0 and 1 for its given inputs.
We can calculate the change in the output using the following:
$$\Delta output \approx \sum_j \frac{\partial output}{\partial w_j}\Delta w_j + \frac{\partial output}{\partial b}\Delta b$$
This means that the change in the output is a linear function of the changes in the weights and bias. This means we can easily choose small changes in these parameters to achieve desired small changes in the output.

The chapter then discusses the architecture of neural networks and talks about a 3 layer system with an input layer, one output neuron and a 'hidden layer'.

It goes on to talk about how the weighting effects the the output of the neurons and how we can use this to train it to recognise 'parts' of a number.

The project will be using the [MNIST data set](http://yann.lecun.com/exdb/mnist/) for its training and testing data.

The training input will be denoted as x, a 28 x 28 = 784-dimensional vector (representing the 784 pixels of the image). The desired output will be denoted by y=y(x). If x is a training image depicting a 6 then $$y(x)=(0,0,0,0,0,0,1,0,0,0)^T$$
We would like an algorithm to tell us what weights and biases will give an output that approximate y(x). We do this by defining a cost function:
$$C(w,b)\equiv \frac{1}{2n} \sum_x ||y(x)-a||^2$$
where w is the collection of all weights in the network, b is the biases, n is the number of training inputs, a is the vector of outputs from the network when x is input and the sum is over all training inputs, x.

This is known as the **quadratic cost function** or the mean squared error (MSE). It becomes small when y(x) approximates the output a. So we would like to minimise the cost function. We do this with an algorithm known as **gradient descent**.

To develop the gradient descent, we will imagine we are given a function of many variables and we want to minimize that function. We will call this C(v) where $v=v_1, v_2, ...$  

We will think of our function as a valley. If we imagine a ball rolling down it then we can imagine it rolling to a minimum. If we move the ball a small amount $\Delta v_1$ in the $v_1$ direction and a small amount $\Delta v_2$ in the $v_2$ direction then we have the following: 
$$\Delta C \approx \frac{\partial C}{\partial v_1}\Delta v_1 + \frac{\partial C}{\partial v_2}\Delta v_2 $$
We will choose $\Delta v1$ and $\Delta v2$ to make $\Delta C$ negative. We will define the gradient of C to be the vector of partial derivatives: 
$$\nabla C \equiv (\frac{\partial C}{\partial v_1},\frac{\partial C}{\partial v_2})^T$$

We can rewrite $\Delta C$ as: 
$$\Delta C \approx \nabla C \cdot \Delta v$$
So if we choose $\Delta v$ to make $\Delta C$ negative, such as:
$$\Delta v = -\eta\nabla C$$
Where $\eta$ is a small, positive parameter known as the learning rate. 

So we will move the ball by the following amount:
$$v \rightarrow v' = v-\eta\nabla C$$
And we keep doing this until we reach a minimum. Often we will vary $\eta$ based on how much the previous iteration changed the cost function C.

To apply this to our weights and bias's we get the following:

$$ w_k \rightarrow w'_k = w_k - \eta\frac{\partial C}{\partial w_k} $$
$$ b_l \rightarrow b'_l = b_l - \eta\frac{\delta C}{\delta b_l} $$
These two equations allow us to repeatedly move 'downhill' on our cost function until we reach a minimum. 

To speed up the calculation of $\nabla C$ we will use stochastic gradient descent. This works by picking out a small number $m$ of randomly chosen training inputs as a mini-batch. This will be used to approximate the average value of $\nabla C$. 
$$\frac{\sum^m_{j=1}\nabla C_{X_j}}{m}\approx \frac{\sum_x\nabla C_x}{n}=\nabla C$$
Where the second sum is over the entire set of training data. Rearranging we get: 
$$\nabla C \approx \frac{1}{m}\sum^m_{j=1}\nabla C_{X_j}$$
Adapting this to our weights and bias's: 
$$ w_k \rightarrow w'_k = w_k - \frac{\eta}{m}\sum_j\frac{\partial C_{X_j}}{\partial w_k} $$
$$ b_l \rightarrow b'_l = b_l - \frac{\eta}{m}\sum_j\frac{\partial C_{X_j}}{\partial b_l} $$
For a third layer, where a is the vector of activations in the second layer, its vector of activations is: 
$$a' = \sigma(wa+b)$$
## Chapter 2

This chapter discusses how to quickly calculate the gradient of the cost function using a technique called backpropagation.

This chapter will use a new notation to refer to weights. $w^l_{jk}$ refers to the connection from the $k^{th}$ neuron in the $(l-1)^{th}$ layer to the $j^{th}$ neuron in the $l^{th}$ layer. Similarly, for a and b we use $b^l_j$ to denote the bias for the $j^{th}$ neuron in the $l^{th}$ layer and $a^l_j$ for the activation of the  $j^{th}$ neuron in the $l^{th}$ layer.

From these, we can calculate the activation $a^l_j$ of the $j^{th}$ neuron in the $l^{th}$ layer using: 
$$a^l_j=\sigma(\sum_kw^l_{jk}a^{l-1}_k+b^l_j)$$
To rewrite this this in matrix form we define matrices $w^l$, $b^l$ and $a^l$ for each layer, $l$. With these matrices we can write the activation vector of a layer in terms of the activation of the previous layer, the weight vector and the bias vector as follows: 
$$a^l=\sigma(w^la^{l-1}+b^l)$$
This equation also computes the intermediate quantity $z^l\equiv w^la^{l-1}+b^l$ along the way. This quantity is useful and is named the weighted input to the neurons in layer $l$.

The goal of backpropagation is to compute the partial derivatives $\delta C/\delta w$ and $\delta C /\delta b$ of the cost function $C$ with respect to any weight or bias in the network. We need to make two main assumptions about the form of the cost function in order to do this. 

First we need to assume that the cost function can be written as an average $C=\frac{1}{n}\sum_xC_x$ over cost functions $C_x$ for individual training examples, $x$. 

The second assumption is that the cost function can be written as a function of the outputs from the neural network: $C = C(a^L)$ 

The backpropagation algorithm uses the Hadamard product, which is the elementwise product of two vectors of the same dimension, denoted by $s\odot t$. 

Backpropagation has four fundamental equations it uses to calculate the gradient of the cost function. We first introduce a quantity $\delta^l_j$ which denotes the $error$ in the $j^{th}$ neuron in the $l^{th}$ layer. 

To understand this error we imagine a demon sitting on the $j^{th}$ neuron in layer $l$. As the input comes in, the demon adds a little change $\Delta z^l_j$ to the neuron's weighted input, so that instead of outputting $\sigma(z^l_j)$, it instead outputs $\sigma(z^l_j+\Delta z^l_j)$. This change propagates through the layers and causes the overall cost to change by an amount $\frac{\partial C}{\partial z^l_j}\Delta z^l_j$. 

Suppose this demon is trying to help and changing the input to make the cost smaller. If $\frac{\partial C}{\partial z^l_j}$ is large, then the demon can lower the cost quite a bit by choosing $\Delta z^l_j$ with the opposite sign. However if it is small then the demon cannot make much of a difference to the overall cost. Therefore, this derivative is a measure of the error in the neuron, so we will define: 
$$ \delta^l_j\equiv\frac{\partial C}{\partial z^l_j}$$
Backpropagation will let us compute $\delta^l$ for each layer, then relating those errors to the partial derivatives of $C$.

The first equation is for the error in the output layer, $\delta^L$:
$$\delta^L=\frac{\partial C}{\partial a^L_j}\sigma'(z^L_j)$$
In matrix form: 
$$\delta^L=\nabla_aC\odot\sigma'(z^L)$$

For a quadratic cost function we have $\nabla_aC=(a^L-y)$ giving:
$$\delta^L=(a^L-y)\odot\sigma'(z^L)$$
The second equation is the error $\delta^l$ in terms of the error in the next layer, $\delta^{l+1}$:
$$\delta^l=((w^{l+1})^T\delta^{l+1})\odot\sigma'(z^l)$$
We can use these two equations to compute the error for every layer in the system. 

The third equation gives the rate of change of the cost w.r.t the bias in the network: 
$$\frac{\partial C}{\partial b^l_j}=\delta^l_j$$
or the shorthand: 
$$\frac{\partial C}{\partial b}=\delta$$
The final equation gives the rate of change of the cost w.r.t the weights in the network: 
$$\frac{\partial C}{\partial w^l_{jk}}=a^{l-1}_k\delta^l_j$$
or the shorthand: 
$$\frac{\partial C}{\partial w}=a_{in}\delta_{out}$$
We will now put these equations into a more explicit algorithm:

1. **Input $x$:** Set the corresponding activation $a^1$ for the input layer.
2. **Feedforward:** For each $l=2,3,...,L$ compute $z^l=w^la^{l-1}+b^l$ and $a^l=\sigma(z^l)$.
3. **Output error $\delta^l$:** Compute the vector $\delta^L=\nabla_aC\odot\sigma'(z^L)$.
4. **Backpropagate the error:** For each $l=L-1, L-2,...,2$ compute $\delta^l=((w^{l+1})^T\delta^{l+1})\odot\sigma'(z^l)$.
5. **Output:** The gradient of the cost function is given by $\frac{\partial C}{\partial w^l_{jk}}=a^{l-1}_k\delta^l_j$ and $\frac{\partial C}{\partial b^l_j}=\delta^l_j$.
