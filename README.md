# pytorch-rbm-autoencoder
An deep autoencoder initialiezd with weights from pre-trained Restricted Boltzmann Machines (RBMs). This implementation is based on the greedy pre-training strategy described by Hinton and Salakhutdinov's paper "[Reducing the Dimensionality of Data with Neural Networks](http://www.cs.toronto.edu/~hinton/science.pdf)" (2006).

This implementation provides support for CPU and GPU (CUDA). Simliar to the original paper, the RBM uses Contrastive Divergence learning for weight updates as described in [this paper](https://christian-igel.github.io/paper/TRBMAI.pdf) rather than pytorch's native optimizers.

## Initializing a Deep Autoencoder (DAE) with Pre-trained RBMs Can Give Better Results
The following images show the reconstructed MNIST images from a 784-1000-500-250-2 autoencoder based on different training strategies. You can see that the RBM pre-training strategy provides better results and a 20% lower loss. 

| Original Image | DAE Naive Training | DAE Initialized with Pretrained RBMs |
| :-----------: | :-----------: | :-----------: |
| ![original image](/images/original_digits.png?raw=true) MSE loss: N/A | ![reconstructed image from naive training](/images/reconstructed_digits_naive_dae.png?raw=true) MSE loss: 0.0674 | ![reconstructed image from pre-trained RBM training](/images/reconstructed_digits_dae.png?raw=true) MSE loss: 0.0303 |


This trend can also be seen when we plot the 2d representations learned by the autoencoders. We also provide a PCA decomposition for comparison.

| PCA | DAE Naive Training | DAE Initialized with Pretrained RBMs |
| :-----------: | :-----------: | :-----------: |
| ![2d representation from PCA](/images/pca_repr.png?raw=true) | ![2d representation from naive training](/images/naive_dae_repr.png?raw=true) | ![2d representation from pre-trained RBM training](/images/dae_repr.png?raw=true) |



## Why Pre-training Helps
A naive training of a deep autoencoder easily gets stuck in a local minimum based on the initialization of the parameters (see amorphous "digit" it learned above in naive training). To fight this, we pre-train RBMs and use the weights from the pretrained RBMs to provide the autoencoder with a good initial state. This good initial state allows the autoencoder to find a good minimum in fine tuning.

## Training Procedure
If you want to train a 784-1000-500-250-2 dimensional autoencoder, pretrain one RBM for each pair of dimensions: 784-1000, 1000-500, 500-250, and 250-2. We use contrastive divergence learning for weight updates and for the final layer, we make the hidden state a Gaussian distribution rather than a Bernoulli distribution to help the final layer take advantage of continuous features of the hidden state. 


Take each of the pre-trained RBMs and stack them to create a deep autoencoder. Each RBM will show up twice in the autoencoder, once as an encoder, and once as a decoder. Finally, fine-tune the autoencoder with more conventional pytorch training methods (stochastic gradient descent with an mean-squared error loss).
