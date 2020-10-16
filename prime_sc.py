import math
import multiprocessing as multi
from timeit import default_timer as timer

def isPrime(num):
    if num <= 1: return False
    if num == 2: return True
    if num % 2 == 0: return False

    boundary = math.floor(math.sqrt(num))

    for x in range(3, boundary + 1, 2):
        if num % x == 0:
            return False
    return True

def findPrimesParallel(numMax):
    full_list = dict()

    for x in range(0, numMax):
        if isPrime(x):
            full_list[x] = True
    return full_list

start = timer()
numMax = 1000000

result = findPrimesParallel(numMax)
tt = timer() - start
print("Number of primes: %d" %len(result))
print("Process time: %f s" %tt)