import numpy as np
from pylab import imshow, show
from timeit import default_timer as timer

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

def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    for x in range(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            imag = min_y + y * pixel_size_y
            image[y, x] = mandel(real, imag, iters)

image = np.zeros((4096, 6144), dtype = np.uint8)

start = timer()
create_fractal(-2.0, 1.0, -1.0, 1.0, image, 50)
tt = timer() - start

print("Total time: %f s" %tt)
imshow(image, cmap='gist_ncar_r')
show()