import math
import multiprocessing as multi
from timeit import default_timer as timer

def findPrimes(first, last):
    def isPrime(num):
        if num <= 1: return False
        if num == 2: return True
        if num % 2 == 0: return False

        boundary = math.floor(math.sqrt(num))

        for x in range(3, boundary + 1, 2):
            if num % x == 0:
                return False
        return True

    primes = dict()
    for num in range(first, last):
        if isPrime(num):
            primes[num] = True
    return primes

def findPrimesParallel(numMax):
    pool = multi.Pool(multi.cpu_count())
    chunk_size = int(numMax / multi.cpu_count())
    results = []
    full_list = dict()

    for x in range(0, numMax, chunk_size):
        end = x + chunk_size
        results.append(pool.apply_async(findPrimes, args=(x, end)))
    pool.close()
    
    for res in results:
        section = res.get()
        full_list = {**full_list, **section}
    full_list = sorted(full_list)
    return full_list

if __name__ == "__main__":
    start = timer()
    numMax = 1000000

    result = findPrimesParallel(numMax)
    tt = timer() - start
    print("Number of primes: %d" %len(result))
    print("Process time: %f s" %tt)