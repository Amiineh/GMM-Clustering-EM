import numpy as np
import random
import math
import matplotlib.pyplot as plt
import sys

def p(x, meu, Sigma):
    return math.exp((-0.5) * np.array(x - meu) * np.linalg.inv(np.matrix(Sigma))* np.array(x - meu).T) / math.sqrt(pow(2 * math.pi, len(x)) * np.linalg.det(np.matrix(Sigma)))

def Cal_w(k, W, X, alfa, Sigma, Meus):

    for i in range(len(X)):
        sum = 0
        for j in range(k):
            sum += float(p(X[i], Meus[j], Sigma[j])) * alfa[j]
            W[i][j] = (float(p(X[i], Meus[j], Sigma[j])) * alfa[j])
        if(sum == 0):
            for j in range(k):
                W[i][j] /= 1./k
        else:
            for j in range(k):
                W[i][j] /= sum

def Cal_alfa(n, k, alfa, W):
    for i in range(k):
        sum = 0
        for j in range(n):
            sum += W[j][i]
        alfa[i] = sum / n

def Cal_meus(n, k, alfa, w, x, meus):
    for i in range(k):
        sum = 0
        for j in range(n):
            sum += np.dot(w[j][i], x[j])
        meus[i] = sum / (alfa[i] * n)

def Cal_sigma(n, k, alfa, w, x, meus, sigma):
    for i in range(k):
        sum = 0
        for j in range(n):
            sum += np.dot(w[j][i], np.matrix(x[j] - meus[i]).T * np.matrix(x[j] - meus[i]))
        sigma[i] = sum / ( alfa[i] * n )

def Find_cluster(w):
    max = 0.
    index = 0
    for i in range(len(w)):
        if w[i] > max:
            max = w[i]
            index = i
    return index

# 0. Getting data
X = []

address = input("Please enter the address of you data:\n")
with open(address) as f:
    for line in f:
        X.append(line.split())

for str in X:
    for i in range(len(str)):
        str[i] = float(str[i])

X = np.matrix(X)
d = len(str)        # d is the dimension of out data (usually chosen 2 to be better ploted)

# 1. Initialize by randomly selecting K mean vectors
k = int(input("How many clusters do you require? (If not sure, run the program once for one cluster to get a better intuition of your data)\n"))
    # k is the nuber of clusters
Meus = []
Sigma = []
W = [[1./k]*k for j in range(len(X))]
alfa = []

for i in range(k):
    r = random.randint(0, len(X) - 1)
    Meus.append(X[r])
    Sigma.append(np.identity(d))
    alfa.append(1./k)

# EM
for i in range(int(25000 / X.size)): # the number of times we should calculate the EM, 25000 is obtained by multiple testcases
    Cal_w(k, W, X, alfa, Sigma, Meus)
    Cal_alfa(len(X), k, alfa, W)
    Cal_meus(len(X), k, alfa, W, X, Meus)
    Meus = np.reshape(Meus, (k, d))
    Cal_sigma(len(X), k, alfa, W, X, Meus, Sigma)
    Sigma = np.reshape(Sigma, (k, d, d))

# ploting
cluster = [[] for i in range(k)]

for i in range(len(X)):
    index = Find_cluster(W[i])
    cluster[index].append([X.item(i, 0), X.item(i, 1)])

for i in range(k):
    tmpX = []
    tmpY = []
    for j in range(len(cluster[i])):
        tmpX.append(cluster[i][j][0])
        tmpY.append(cluster[i][j][1])
    plt.plot(tmpX, tmpY, 'o')

plt.plot(np.array(Meus).T[0], np.array(Meus).T[1], '^')
plt.show()

