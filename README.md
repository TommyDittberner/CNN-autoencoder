# CNN Autoencoder
A convolutional neural network autoencer written in Python (v.3.6) with Tensorflow (v.1.13) that uses the "Stanford Dogs" Dataset

To run this project yourself you need to have **tensorflow, Pillow (PIL), matplotlib and numpy** installed

## What it does
An autoencoder compresses images down to very few numbers and then tries to recreate the original image from the information that is still left. For some images this works better than for others. (They need to be somewhat similar.)

The images get scaled down to 32x32 images before being fed into the cnn-autoencoder which compresses them from further down into 4x4 images and back. Convolutions are 5x5 pixels and in the last layer there are 24 filters in use.

The project plots the original images and their recreation into a PDF file, as well as the loss per epoch and the best and worst recreations overall. To get even better results you can try one of those things: 
- Increase the number of epochs to 50, 100, 200 or even more (this will take a while to run!) 
- Increase the scaled down size of the images to 64x64 instead of 32x32 (might require some work though)
- Increase the size of the convolutions to a bigger number


## What I've learned from this project
- how to use tensorflow
- how to prepare data to feed it into a neural network
- how convolutions work
- how to plot results using matplotlib
