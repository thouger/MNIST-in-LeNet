![LeNet-5 Architecture](lenet.png)
Implement the LeNet-5 deep neural network model.

Specs
Convolution layer 1. The output shape should be 28x28x6.

Activation 1. Your choice of activation function.

Pooling layer 1. The output shape should be 14x14x6.

Convolution layer 2. The output shape should be 10x10x16.

Activation 2. Your choice of activation function.

Pooling layer 2. The output shape should be 5x5x16.

Flatten layer. Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. The easiest way to do is by using tf.contrib.layers.flatten, which is already imported for you.

Fully connected layer 1. This should have 120 outputs.

Activation 3. Your choice of activation function.

Fully connected layer 2. This should have 10 outputs.

Return the result of the 2nd fully connected layer from the LeNet function.
