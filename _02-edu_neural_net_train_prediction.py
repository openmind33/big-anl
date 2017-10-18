"""
Title : Numpy Array
Program : code name : numpy-array.py 
Autho : 이이백(yibeck.lee@gmail.com)
"""

import numpy as np 

print(__doc__)

# 3 * 10 array 생성
arr_3by10 = np.array([
	[1,1,1,1,1,1,1,1,1,1],
	[1,1,1,1,1,2,2,2,2,2],
	[1,2,3,4,5,6,7,8,9,10],
	])
print(arr_3by10)

print(arr_3by10.shape)

print('nrow = ', arr_3by10.shape[0])
print('ncol = ', arr_3by10.shape[1])

# 배열의 합
print(np.sum(arr_3by10))

# 배열의 합 - column 기준
print(np.sum(arr_3by10, axis=0))

# 배열의 합 - row 기준
print(np.sum(arr_3by10, axis=1))


# 배열의 평균
print(np.mean(arr_3by10))

# 배열의 평균 - column & row
print(np.mean(arr_3by10, axis=0)); print(np.mean(arr_3by10, axis=1))
# 배열의 표준편차
print(np.std(arr_3by10))
print(np.std(arr_3by10, axis=0))
print(np.std(arr_3by10, axis=1))
