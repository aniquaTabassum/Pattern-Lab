
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

mean_of_class_a = np.array([0,0])
mean_of_class_b = np.array([0,0])
def find_class(point = np.array([])):
    global mean_of_class_a
    global mean_of_class_b
    linear_discriminant1 = (np.matmul(point.transpose(),mean_of_class_a)) + 0.5*(np.matmul(mean_of_class_a.transpose(),mean_of_class_a))
    print(linear_discriminant1)
test_set = []
train= []
with open('/home/aniqua/Downloads/test.csv', 'r') as file:
    my_reader = csv.reader(file, delimiter=' ')
    for row in my_reader:
        test_set.append(row)

with open('/home/aniqua/Downloads/train.csv', 'r') as file:
    my_reader = csv.reader(file, delimiter=' ')
    for row in my_reader:
        train.append(row)

for i in range(len(test_set)):
    for j in range(len(test_set[i])):
        test_set[i][j] = int(test_set[i][j])

for i in range(len(train)):
    for j in range(len(train[i])):
        train[i][j] = int(train[i][j])
print(train)
train_inter = []
for i in range(len(train)):
    for j in range(3):
        train_inter[i][j] = int(train[i][j])
train_set = np.array(train_inter)

a = [[t[0], t[1]] for t in test_set if t[2] == 1]
b = [[t[0], t[1]] for t in test_set if t[2] == 2]
class_a = np.array(a)
class_b = np.array(b)
#print(class_a)
plt.plot(class_a[:, 1:],class_a[:,0:1], linestyle = '', marker='o', color='k')
plt.plot(class_b[:, 1:], class_b[:,0:1], linestyle = '', marker = 'o', color = 'r')
plt.show()


mean_of_class_a[0]  = np.mean(class_a[:,0:1])
mean_of_class_a[1] = np.mean(class_a[:,1:])



mean_of_class_b[0]  = np.mean(class_b[:,0:1])
mean_of_class_b[1] = np.mean(class_b[:,1:])

for i in range(len(train_set)):
    find_class(train_set[i])