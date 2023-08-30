import numpy as np


class LossFuncs:
    def __init__(self):
        self.epsilon = 1e-15

    def __call__(self, func, *args):  # to make the object as callable function

        result = getattr(self, func)(*args)
        return result

    def MSE(self, a_n, y):
        m = y.size
        cost = np.sum((a_n - y) ** 2)
        return (1 / (2 * m)) * cost

    def dMSE(self, a_n, y):
        return a_n - y

    def CE(self, a_n, y):
        ce = -1 * (y @ np.log(a_n))
        return ce

    def Batch_CE(self, y, a_n):
        # to avoid of taking the logarithm of zero, we add s small constant epsilon 1e-15
        ce = np.sum(np.multiply(y, np.log(a_n + self.epsilon)))
        l_ind = np.sum(-np.multiply(y, np.log(a_n + self.epsilon)), axis=1)  # return a vector of size (y.shape[0],1)
        l = -(1. / y.shape[0]) * ce
        return l, l_ind

    def dCE(self, y, a_n):  # derivative of cross entropy loss with softmax
        d = (a_n - y)
        return d


# ---------------------------------------------------------------------------------------
class ActivationFuncs:

    def __init__(self):
        pass

    def __call__(self, func, *args):  # to make the object as callable function the same object
        result = getattr(self, func)(*args)
        return result

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def dsigmoid(self, z):  # the  derivative of sigmoid function
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def ReLU(self, z):
        return np.maximum(0, z)

    def dReLU(self,
              z):  # the derivative of ReLU function using numpy.where() to replace elements greater than or equal 0 with 1 and others with 0
        return np.where(z <= 0, 0, 1)

    def softmax_b(self, z):
        # to avoid numerical Instability, the maximum element in Z is subtracted from all elements
        e = np.exp(z - np.max(z, axis=1, keepdims=True))

        return e / np.sum(e, axis=1, keepdims=True)
        # return np.exp(z)/np.sum(np.exp(z),axis=0)

    def softmax(self, z):
        # to avoid numerical Instability, the maximum element in Z is supstracted from all elements
        e = np.exp(z - np.max(z))
        return e / np.sum(e, axis=0)
