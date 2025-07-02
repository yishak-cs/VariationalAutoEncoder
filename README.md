# Variational Autoencoder (VAE) for MNIST Digit Generation

This repository contains a PyTorch implementation of a Variational Autoencoder (VAE) that learns to generate handwritten digits from the MNIST dataset. The model compresses the high-dimensional image data into a continuous, 2D latent space and then reconstructs the original images, enabling the generation of new, plausible digits.

## What is an Autoencoder?

Autoencoders are self-supervised neural networks designed for dimensionality reduction. Their primary goal is to first compress (encode) input data into a low-dimensional representation and then reconstruct (decode) the original data from this compressed form with minimal loss.

At its core, an autoencoder learns to extract the most salient features—the latent variables—while discarding irrelevant noise. Different types of autoencoders employ unique strategies for this information extraction, making them suitable for various use cases.

All autoencoders share three fundamental components:

The Encoder → The Latent Space → The Decoder

### The Problem with Regular Autoencoders

While regular autoencoders are effective at compressing data, their latent space is often non-regularized and discontinuous. This means the space can have "gaps" or "holes" where decoded outputs are not meaningful. Consequently, you cannot sample a random point from the latent space and expect the decoder to generate a valid output.

### What Makes VAEs Different?

Variational Autoencoders solve this problem by introducing a probabilistic approach. Instead of encoding an input as a single, discrete point in the latent space, a VAE encodes it as a probability distribution, typically a Gaussian. It learns two vectors: a mean (mu) and a log-variance (logsigma2).

This approach forces the latent space to be continuous and dense, a property reinforced by a technique called regularization.

### The Power of Regularization

Regularization is a technique used to force the latent space to be more organized and meaningful.

    The Problem (Un-regularized Space): A regular autoencoder learns to cluster digits effectively but leaves empty, meaningless regions between clusters. You can't pick a random point and expect a valid output.

    The Solution (Regularized Space): The VAE's loss function includes a regularization term that encourages the encoder to produce distributions that are close to a standard normal distribution. This pushes the clusters closer together and creates a smooth latent space where transitions between digits are meaningful. The goal is to make every point in the latent space correspond to a valid-looking output when decoded.

Model Architecture

1. Encoder Network

The encoder compresses the input images into a 2D probabilistic latent representation.

    Layers: 2 fully connected layers (784 → 512 → 512)

    Activation Function: LeakyReLU (alpha=0.2)

    Output: Two separate fully connected layers to produce the mean (mu) and log-variance (logsigma2) vectors for the 2D latent space.

2. Decoder Network

The decoder reconstructs the 28x28 pixel images from the 2D latent vectors.

    Layers: 3 fully connected layers (2 → 512 → 512 → 784)

    Activation Functions: LeakyReLU for the hidden layers and a Sigmoid function for the output layer to scale pixel values between 0 and 1.

3. Reparameterization Trick

Instead of sampling directly from the learned distribution, we sample from a standard normal distribution and then scale and shift it using the learned mean and variance.

The latent vector z is computed as:
z=μ+σ⋅ϵ, where ϵ∼N(0,1)

### Loss Function

The VAE loss function is a combination of two distinct components:

    Reconstruction Loss (Binary Cross-Entropy): This loss measures how accurately the decoder reconstructs the original input image. It ensures that the generated images are visually similar to the originals.

    KL Divergence Loss (Regularization): This loss acts as a regularization mechanism. It forces the learned latent distribution to be close to a standard normal distribution (mathcalN(0,1)), creating the smooth and continuous latent space that makes VAEs so powerful. The formula is:
    $$$$$$L\_{KL} = -0.5 \\cdot \\sum \\left(1 + \\log(\\sigma^2) - \\mu^2 - \\sigma^2\\right) $$

    $$$$
    $$The total loss is the sum of these two losses.

### Training Process

The model is trained using the Adam optimizer and standard backpropagation.

## Visualizations

This project includes three key visualization methods to demonstrate the VAE's capabilities.

1. Digit Generation

Generate new, unique digits by sampling specific coordinates from the 2D latent space. This showcases the model's generative power.

    Input: An (x, y) coordinate in the latent space.

    Output: A generated 28x28 digit image.

### 2. Latent Space Visualization

Create a comprehensive 15x15 grid that maps the entire 2D latent space (from coordinates -2 to +2). This reveals how different digit classes are organized and shows the smooth, continuous transitions between them.

### 3. Reconstruction Comparison

Evaluate the model's ability to preserve key visual features by comparing original test images with their reconstructed counterparts.

    Top Row: Original MNIST test images.

    Bottom Row: VAE reconstructions from the latent space.
