import numpy as np
import math

def get_noise_sum(input_size, epsilon, test_size=10000):  
    noise_list = np.random.laplace(0, np.log2(input_size)/epsilon, (test_size, math.floor(np.log2(input_size))))
    return np.sum(noise_list, axis=1)

def noise_sum_accuracy():
    test_size = 1000000
    noise_sum_list = get_noise_sum(100000, 10, test_size)
    print(noise_sum_list)
    for cond in range(1, 1000):
        delta = np.count_nonzero((noise_sum_list>cond) | (noise_sum_list<-cond))/test_size
        print(cond, delta)

noise_sum_accuracy()