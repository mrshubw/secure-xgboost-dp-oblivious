import math

def get_private_memory_size(input_size, epsilon, delta, s=0):
    if 0!=s:
        input_size = input_size/s
    return math.pow(math.log2(input_size), 1.5)*math.log(2*input_size/delta)*math.sqrt(8)/epsilon

def get_privacy(input_size, delta, private_size):
    return math.pow(math.log2(input_size), 1.5)*math.log(2*input_size/delta)*math.sqrt(8)/private_size

# s=0
# for _ in range(50):
#     s = get_private_memory_size(16110, 1, 0.00001, s)
#     print(s)

dataset_size_list = [1e3, 1e4, 1e5]
private_mem_size_list = [2**i for i in range(6, 17)]
delta = 1e-5
epsilon = [0.1, 1, 10]

print("compute epsilon:")
privacy = []
for dataset_size in dataset_size_list:
    print("dataset size:", dataset_size)
    temp = []
    for private_mem_size in private_mem_size_list:
        print("private memory list:", private_mem_size)
        s = get_privacy(dataset_size, delta, private_mem_size)
        print(s)
        temp.append(s)
    privacy.append(temp)

print("*"*50)
print("compute private memory size:")
privacy = []
for dataset_size in dataset_size_list:
    print("dataset size:", dataset_size)
    temp = []
    for eps in epsilon:
        print("eps:", eps)
        s = get_private_memory_size(dataset_size, eps, delta)
        print(s)
        temp.append(s)
    privacy.append(temp)