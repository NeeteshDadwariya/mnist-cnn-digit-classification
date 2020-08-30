The project aims to natively classifies hand written images from the MNIST 784 dataset.

# About the dataset
The MNIST data set contains 70000 images of handwritten digits. This is perfect for anyone who wants to get started with image classification as a beginner. This is because, the set is neither too big to make beginners overwhelmed, nor too small so as to discard it altogether.

# Training Pipeline
Training pipeline can be seen using the [notebook.ipynb](notebook.ipynb).

For training the neural network, I've used stochastic gradient descent, which means we put one image through the neural network at a time.

## Forward Pass
The forward pass consists of the dot operation in NumPy, which turns out to be just matrix multiplication. As described in the introduction to neural networks article, we have to multiply the weights by the activations of the previous layer. Then we have to apply the activation function to the outcome.

## Layers
The network contains one input layer and 2 hidden layers to reduce the image dimenion to 1x10 scale so as to be applied to the activation function.

## Activation function
The last layer constitutes of the softmax-activation function as this returns a probability distribution over the target classes in a multiclass classification problem. In our case, the activation classes are limited to 0-9 each representing one digit. 

# Inference Pipeline
For the inference pipeline, streamlit web application has been et up.
Streamlit web app can be run via the below command

`pip install streamlit`

`streamlit run webapp.py`

Below is the working screenshot of the webapp:

![Inference Pipeline](/images/streamlit-inference-pipeline.png?raw=true)
