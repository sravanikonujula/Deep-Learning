import tensorflow as tensef1
import numpy as np
from sklearn.datasets import load_breast_cancer
from Q1 import pred1
# Considering a data set and loading it
dset1 = load_breast_cancer()
data = dset1.data
labels = dset1.target

#taking the labels in array format
#569 rows
labels = np.array(labels).reshape(569,1)

#Constructing a placeholderds for 30 columns
X1 = tensef1.placeholder(tensef1.float32, shape=[None, 30])
y1 = tensef1.placeholder(tensef1.float32, shape=[None, 1])
# goes with random values.
tensef1.set_random_seed(1)
#weights getting started
W3 = tensef1.Variable(tensef1.zeros([30, 1]))
b1 = 0
#operational functions
Activity1 = tensef1.nn.sigmoid(tensef1.add(tensef1.matmul(X1, W3), b1))

#It is for loss
loSS1 = tensef1.reduce_mean(tensef1.nn.sigmoid_cross_entropy_with_logits(logits=Activity1, labels=y1))
#optimixer for gradient descent
optimaL1 = tensef1.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loSS1)
# Declaring global variables
init = tensef1.global_variables_initializer()
# Session method declaration
with tensef1.Session() as sess:
    sess.run(init)
    writer = tensef1.summary.FileWriter("./graphs/logistic_reg", sess.graph)
#considering range for 200 values
    for k in range(300):

        _, acc1 = sess.run([optimaL1, loSS1], feed_dict={X1:data, y1:labels})

        if k%20==0:
            print("cost: " + str(acc1))

    writer.close()
    parameters = sess.run(W3)
print("Done with the Optimization!")
outs = pred1(data, parameters) #Calling the function
#Calculating accuracy score for the regression model
A=format(100 - np.mean(np.abs(outs - labels)) * 100)

print("Required Accuracy is :")
print(A)