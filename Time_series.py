import statistics
import numpy as np

def moving_ave(arr, n):
    result = [None] * n
    for i in range(len(arr)-(n-1)):
        result.append(statistics.mean(arr[i:i+n]))
    print(result)

def weighted_moving_ave(arr, n):
    result = [None] * n
    for i in range(len(arr)-(n-1)):
        result.append(sum(arr[i:i+n] * np.array(range(1, n+1)) / \
                          sum(range(n+1))))
    print(result)
 
arr = [7, 14, 11, 19, 9, 8, 12, 11, 7, 10, 10]
time_period = 3

moving_ave(arr, time_period)
weighted_moving_ave(arr, time_period)