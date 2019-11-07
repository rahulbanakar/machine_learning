import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

def segmoid(z):
    return (1/(1 + np.exp(-z)))

def costFUnction(w,x,y,lamb):
    m = len(y)
    y = y[:,np.newaxis]
    value = segmoid(x @ w)
    error = (-y * np.log(value)) - ((1-y) * np.log(1-value))
    cost = 1/m * sum(error)
    regcost = cost + lamb/(2*m) * sum(w**2)

    #compute gradient
    w0 = 1/m * (x.transpose()  @ (value - y))[0]
    w1 = (1/m * (x.transpose() @ (value - y))[1:] + (lamb/m)*w[1:])
    #grad = np.vstack((w0[:,np.newaxis], w1))
    print(w0)
    print(w1)
    grad = np.vstack((w0[:,np.newaxis],w1))
    return regcost[-1],grad

def gradientDescent(x,y,w,alpha,iter,lamb):
    m = len(y)
    w_history = []
    for i in range(0,iter):
        cost,grad = costFUnction(w,x,y,lamb)
        w = w - (alpha * grad)
        w_history.append(cost)
    return w,w_history

def predict(w,x):
    Predict = x @ w
    return Predict>0


# read files
D_tr = genfromtxt('C:/Rahul/suma/Programming/spambasetrain.csv', delimiter = ',', encoding = 'utf-8')
D_ts = genfromtxt('C:/Rahul/suma/Programming/spambasetest.csv', delimiter = ',', encoding = 'utf-8')

# construct x and y for training and testing
X_tr = D_tr[: ,: -1]
y_tr = D_tr[: , -1]
X_ts = D_ts[: ,: -1]
y_ts = D_ts[: , -1]

# number of training / testing samples
n_tr = D_tr.shape[0]
n_ts = D_ts.shape[0]

# add 1 as feature
X_tr = np.concatenate((np.ones((n_tr, 1)), X_tr), axis = 1)
X_ts = np.concatenate((np.ones((n_ts, 1)), X_ts), axis = 1)

#Initialise weight vector
initial_w = np.zeros((X_tr.shape[1],1))
#Regularization parameter
Lambda = 1
cost, grad = costFUnction(initial_w,X_tr,y_tr,Lambda)
print('Initial cost',cost)

#calculate gradientdescent
alpha = 0.1
iter = 1000
Lambda = 0.0001
w,history = gradientDescent(X_tr,y_tr,initial_w,alpha,iter,Lambda)

plt.plot(history)
plt.show()

p = predict(w,X_tr)
print("Train accuraccy : ", ((sum(p==y_tr[:,np.newaxis])/len(y_tr)) * 100)[0],"%")

#test results
test_predict = predict(w,X_ts)
print("Train accuraccy : ", ((sum(test_predict==y_ts[:,np.newaxis])/len(y_ts)) * 100)[0],"%")


print(test_predict)
