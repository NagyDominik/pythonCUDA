import numpy as np
from pylab import imshow, show
from timeit import default_timer as timer
from numba import cuda

@cuda.jit(device=True) #Device funtion
def mandel(x, y, max_iters):
    """
        Given the real and imaginary parts of a complex number,
        determine if it is a candidate for membership in the Mandelbrot
        set given a fixed number of iterations.
    """
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i
    return max_iters

@cuda.jit
def create_fractal(min_x, max_x, min_y, max_y, image, iters): #Kernel
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width 
    pixel_size_y = (max_y - min_y) / height

    startX, startY = cuda.grid(2) #Current thread position
    gridX = cuda.gridDim.x * cuda.blockDim.x
    gridY = cuda.gridDim.y * cuda.blockDim.y
    
    for x in range(startX, width, gridX): #Runs every "gridX"th number starting from startX
        real = min_x + x * pixel_size_x
        for y in range(startY, height, gridY):
            imag = min_y + y * pixel_size_y
            image[y, x] = mandel(real, imag, iters)

image = np.zeros((4096, 6144), dtype = np.uint8) #25 165 824 pixels
griddim = (64, 64) #Max x=2^32-1, Max y=65535
blockdim = (32, 32) #Max x,y=1024

start = timer()
d_image = cuda.to_device(image)
create_fractal[griddim, blockdim](-2.0, 1.0, -1.0, 1.0, d_image, 200)
d_image.to_host()
tt = timer() - start

# start = timer()
# d_image = cuda.to_device(image)
# create_fractal[griddim, blockdim](-2.0, 1.0, -1.0, 1.0, d_image, 200)
# d_image.to_host()
# tt = timer() - start

print("Total time: %f s" %tt)
imshow(image, cmap='gist_ncar_r')
show()