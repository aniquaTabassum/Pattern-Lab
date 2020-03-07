import numpy as np
import matplotlib.pyplot as plt
#a = np.array([1,2,3])
#print(a)
#print(a.shape)
#a[2] = 100
#print(a)
#arrayOfZeros = np.zeros((2,2))
#print(arrayOfZeros)
#arrayOfOnes = np.ones((3,2))
#print(arrayOfOnes)
#arrayOfANumber = np.full((7,2),6)
#print(arrayOfANumber)
#randomNumbers = np.random.random()

#array_of_ones = np.ones([3, 4, 5], dtype=np.float32)
#print(array_of_ones)
#array_of_random = np.random.uniform(1,2,[3,2,4])
#print(array_of_random)
a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
b = a[1:3, 1:3]
print(b)
row_r1 = a[1:2, :]
print(row_r1, np.shape(row_r1))
a = np.array([[1,2,3,4,5,6,7,8],[9,10,11,12,13,14,15,16]])
b = np.array([0,2])
a[np.arange(2),b] += 10
print(a[np.arange(2),b])
print(a)
x = np.arange(0, np.pi*3,0.1)
y = np.sin(x)
print(x)
plt.plot(x,y)
plt.show()