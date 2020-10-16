import numpy as np
import multiprocessing as multi
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

def parallel_iterate(start, end, min_x, min_y, ps_x, ps_y, iters, height):
    ret = np.zeros((4096, 6144), dtype = np.uint8)
    for x in range(start, end):
        real = min_x + x * ps_x
        for y in range(height):
            imag = min_y + y * ps_y
            ret[y, x] = mandel(real, imag, iters)
    return ret

def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    pool = multi.Pool(multi.cpu_count())
    chunk_size = int(width / multi.cpu_count())
    results = []

    for x in range(0, width, chunk_size):
        end = x + chunk_size
        results.append(pool.apply_async(parallel_iterate, args=(x, end, min_x, min_y, pixel_size_x, pixel_size_y, iters, height)))
    pool.close()
    
    for res in results:
        section = res.get()
        image += section

if __name__ == "__main__":   
    image = np.zeros((4096, 6144), dtype = np.uint8)

    start = timer()
    create_fractal(-2.0, 1.0, -1.0, 1.0, image, 50)
    tt = timer() - start

    print("Total time: %f s" %tt)
    imshow(image, cmap='gist_ncar_r')
    show()