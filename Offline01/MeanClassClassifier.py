import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

classification = []
train_set = []
initial_test_set = []
mean_of_class_a = np.array([0,0], dtype=np.float64)
mean_of_class_b = np.array([0,0], dtype=np.float64)


def find_class(point = np.array([], dtype= np.float64)):
    global mean_of_class_a
    global mean_of_class_b
    linear_discriminant1 = (np.matmul(point.transpose(),mean_of_class_a)) + 0.5*(np.matmul(mean_of_class_a.transpose(),mean_of_class_a))
    linear_discriminant2 = (np.matmul(point.transpose(),mean_of_class_b)) + 0.5*(np.matmul(mean_of_class_b.transpose(),mean_of_class_b))
    #print("ld 1 ", linear_discriminant1," ld2 ", linear_discriminant2)
    if(linear_discriminant1 >= linear_discriminant2):
        classification.extend([1])
        return 1
    else:
        classification.extend([2])
        return 2
with open('/home/aniqua/Downloads/train.csv', 'r') as file:
    my_reader = csv.reader(file, delimiter=' ')
    for row in my_reader:
        train_set.append(row)

with open('/home/aniqua/Downloads/test.csv', 'r') as file:
    my_reader = csv.reader(file, delimiter=' ')
    for row in my_reader:
        initial_test_set.append(row)

for i in range(len(train_set)):
    for j in range(len(train_set[i])):
        train_set[i][j] = int(train_set[i][j])

for i in range(len(initial_test_set)):
    for j in range(len(initial_test_set[i])):
        initial_test_set[i][j] = int(initial_test_set[i][j])

a = [[t[0], t[1]] for t in train_set if t[2] == 1]
b = [[t[0], t[1]] for t in train_set if t[2] == 2]
intermediate_test_set = [[t[0], t[1]] for t in initial_test_set]
test_set = np.array(intermediate_test_set)
undivided_test_set = np.array(initial_test_set)
class_a = np.array(a)
class_b = np.array(b)



mean_of_class_a[0]  = np.mean(class_a[:,0:1], dtype=np.float64)
mean_of_class_a[1] = np.mean(class_a[:,1:], dtype=np.float64)
mean_of_class_b[0]  = np.mean(class_b[:,0:1], dtype=np.float64)
mean_of_class_b[1] = np.mean(class_b[:,1:], dtype=np.float64)


for i in range(len(test_set)):
    class_to_plot = find_class(test_set[i])
    if(class_to_plot == 1):
        plt.plot(test_set[i][0],test_set[i][1], linestyle='', marker='x', color='k')
    else:
        plt.plot(test_set[i][0],test_set[i][1],  linestyle='', marker='v', color='r')

midpoint = [(mean_of_class_a[0] + mean_of_class_b[0])/2, (mean_of_class_a[1] + mean_of_class_b[1])/2]
slope = ((mean_of_class_b[1] - mean_of_class_a[1])/(mean_of_class_b[0] - mean_of_class_a[0]))
slope*=-1
slope = 1/slope
constant = midpoint[1] - (slope*midpoint[0])
int_midpoint = int(midpoint[0])

x_values = np.linspace((int_midpoint - 10),10,30)
y_values = [((slope*t) + constant) for t in x_values ]

plt.plot(class_a[:,0:1],class_a[:, 1:], linestyle = '', marker='o', color='k')
plt.plot( class_b[:,0:1],class_b[:, 1:], linestyle = '', marker = 'o', color = 'r')
plt.plot( mean_of_class_a[0], mean_of_class_a[1], linestyle = '', marker='+', color='k')
plt.plot( mean_of_class_b[0], mean_of_class_b[1],linestyle = '', marker='+', color='r')
plt.plot(x_values,y_values, linestyle = '--', color='k')
plt.show()

extract_classes = [t[2] for t in undivided_test_set]
matches = sum([1 for i, j in zip(extract_classes, classification) if i == j])
print("accuracy is ", (matches/len(extract_classes))*100)