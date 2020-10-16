import math
import numpy as np
from numba import cuda
from timeit import default_timer as timer

@cuda.jit(device=True)
def isPrime(num):
    if num <= 1: return False
    if num == 2: return True
    if num % 2 == 0: return False

    boundary = int(num**(1/2))

    for x in range(3, boundary + 2, 2):
        if num % x == 0:
            return False
    return True

@cuda.jit
def findPrimesParallel(numMax, primes):
    start = cuda.grid(1)
    threads = cuda.gridDim.x * cuda.blockDim.x
    for x in range(start, numMax, threads):
        primes[x] = isPrime(x)

numMax = 1000000
primes = np.zeros(numMax, dtype = np.bool)
griddim = 128
blockdim = 1024 # Max 1024

start = timer()
d_primes = cuda.to_device(primes)
findPrimesParallel[griddim,blockdim](numMax, d_primes)
d_primes.to_host()
tt = timer() - start

# start = timer()
# d_primes = cuda.to_device(primes)
# findPrimesParallel[griddim,blockdim](numMax, d_primes)
# d_primes.to_host()
# tt = timer() - start

filtered = np.where(primes == True)[0]
print("Number of primes: %d" %len(filtered))
print("Process time: %f s" %tt)
