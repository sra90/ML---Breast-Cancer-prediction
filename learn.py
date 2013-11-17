#Logistic Regression, gradient descent
import numpy
import scipy.io
import math

def gradientDescent(x,y):
    (nm,nu) = numpy.shape(x)          
    theta =  numpy.mat(numpy.zeros((nu,1)))
    #Currently running gradiescent for 1000 iterations
    for i in range(1,1000):        
        theta = costFunc(x,y,theta)
        
    return theta

def sigmoid(x,theta):
    (m,u) = numpy.shape(x)
    res = numpy.mat(numpy.zeros((m,1)))
    
    res = theta.T * x.T
    res = res.T
    
    for j in range(0,m):
        res[j] = 1/(1+math.exp(-res[j]))
        
    return res

def costFunc(x,y,theta):
    m = 400 #Number of examples in the training set
    sig = sigmoid(x,theta)
        
    #Calculating the cost for current iteration

    #lmbda = 0.000000001
    lmbda = 0 #Regulatization paramater
    reg = (lmbda*(2*400.0))* numpy.sum(numpy.power(theta,2))    #Calculating the regularization term
    ans = numpy.multiply(y , numpy.log(sig)) + numpy.multiply((1-y),numpy.log(1-sig))
    #Cost calculation
    J = (-(1/400.0)*(numpy.sum(ans)))+reg
    print "Cost: ",J
    theta_tmp =  numpy.mat(numpy.zeros((10,1)))
    alpha = 0.1     #Learning rate
    
    #Calculating all theta's for the current iteration of gradient descent
    for i in range(0,10):        
            theta_tmp[i] = ((1/400.0)*numpy.sum(numpy.multiply((sig-y),x[0:m][:,i:(i+1)])))
            if i>0:
                theta_tmp[i] -=((lmbda/400.0)*theta[i])
            theta_tmp[i] = alpha * theta_tmp[i]  

    theta = theta - theta_tmp
    
    return theta
    
def main():
    
    data  = numpy.loadtxt("C:\Rohit\Machine Learning\Cancer Set\data.txt", delimiter=',')
    #Taking 60%,20%,20% of input data for the training,cross validation and test set repectively

    #Training
    print "Training in progress...."
    m = 400
    x = numpy.mat(numpy.ones((m,10)))    #input 
    x[0:m][:,1:10] = numpy.mat(data[0:m][:,1:10])
    y = numpy.mat(data[0:m][:,10:11])       #output

    #Setting y to 0,1 instead of 2,4 in the input dataset
    for k in range(0,m):
        if y[k]==2:
            y[k] = 0
        else:
            y[k] = 1
    
    theta = gradientDescent(x,y)
    print "Calculated theta vector after training"
    print theta

    #Calculating the predicted outputs for the training set using theta learnt from the training set
    a =  sigmoid(x,theta)
    w = 0
    o = 0
    #Counting the number of predictions that are wrong
    for i in range(0,m):
        if a[i]>0.5:
            #print 1, " ",y[i]
            if y[i] != 1:
                w+=1
        else:
            #print 0, " ",y[i]
            if y[i] != 0:
                o+=1
    print "Training set"
    print "Number of 1's that were wrongly predicted :",w
    print "Number of 0's that were wrongly predicted :",o
    print "Training Error:",(1/(2.0*400.0))*numpy.sum(numpy.power((a - y),2))

    #Cross validation
    print "Cross Validation"
    x = numpy.mat(numpy.ones((143,10)))   #input
    x[0:m][:,1:10] = numpy.mat(data[400:543][:,1:10])
    y = numpy.mat(data[400:543][:,10:11])    #output

    #Calculating the predicted outputs for the cross validation set using theta learnt after training
    a =  sigmoid(x,theta)
    w = 0
    o = 0

    #Setting y to 0,1 instead of 2,4 in the input dataset
    for k in range(0,143):
        if y[k]==2:
            y[k] = 0
        else:
            y[k] = 1

    #Counting the number of predictions that are wrong
    for i in range(0,143):
        if a[i]>0.5:
            #print 1, " ",y[i]
            if y[i] != 1:
                w+=1
        else:
            #print 0, " ",y[i]
            if y[i] != 0:
                o+=1
    print "Number of 1's that were wrongly predicted :",w
    print "Number of 0's that were wrongly predicted :",o
    print "Cross validation error : ",(1/(2.0*143.0))*numpy.sum(numpy.power((a - y),2))

    #Test set
    print "Test set"
 
    x = numpy.mat(numpy.ones((140,10))) #input
    x[0:m][:,1:10] = numpy.mat(data[543:683][:,1:10])
    y = numpy.mat(data[543:683][:,10:11]) #output
    #Calculating the predicted outputs for the test set using theta learnt after training
    a =  sigmoid(x,theta)
    w = 0
    o = 0
    #Setting y to 0,1 instead of 2,4 in the input dataset
    for k in range(0,140):
        if y[k]==2:
            y[k] = 0
        else:
            y[k] = 1
    #Counting the number of predictions that are wrong
    for i in range(0,140):
        if a[i]>0.5:
            #print 1, " ",y[i]
            if y[i] != 1:
                w+=1
        else:
            #print 0, " ",y[i]
            if y[i] != 0:
                o+=1
    print "Number of 1's that were wrongly predicted :",w
    print "Number of 0's that were wrongly predicted :",o
    print "Training set error : ",(1/(2.0*140.0))*numpy.sum(numpy.power((a - y),2))
    
main()
