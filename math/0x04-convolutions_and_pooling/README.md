0x04. Convolutions and Pooling

Tasks
-----

#### 0\. Valid Convolution

Function `def convolve_grayscale_valid(images, kernel):` that performs a valid convolution on grayscale images

#### 1\. Same Convolution mandatory

Function `def convolve_grayscale_same(images, kernel):` that performs a same convolution on grayscale images

#### 2\. Convolution with Padding

Function `def convolve_grayscale_padding(images, kernel, padding):` that performs a convolution on grayscale images with custom padding

#### 3\. Strided Convolution

Function `def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):` that performs a convolution on grayscale images

#### 4\. Convolution with Channels

Function `def convolve_channels(images, kernel, padding='same', stride=(1, 1)):` that performs a convolution on images with channels

#### 5\. Multiple Kernels 

Function `def convolve(images, kernels, padding='same', stride=(1, 1)):` that performs a convolution on images using multiple kernels

#### 6\. Pooling 

Function `def pool(images, kernel_shape, stride, mode='max'):` that performs pooling on images