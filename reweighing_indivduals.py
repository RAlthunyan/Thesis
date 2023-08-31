import pickle
import random
from numpy.random import seed
import math

from WeightingStrategies import *
from NNFunctions import *
from keras.datasets import mnist

from batchgenerator import *
import tensorflow as tf
import numpy as np


class Network:
    def __init__(self, structure, Actfun, LossFun, optimizer, learning_rate, batch_size,
                 initialization='He initialization'):
        self.weighting = None
        self.structure = structure
        self.num_layers = len(structure)
        self.Actfun = Actfun
        self.LossFun = LossFun
        self.B_n = [np.zeros((l, 1)) for l in structure[1:]]
        if initialization == 'He initialization':
            self.W_n = [np.random.randn(l, next_l) * np.sqrt(2. / l) for l, next_l in
                        zip(structure[:-1], structure[1:])]  # He initialization "
        elif initialization == 'Glorot_Normal':
            self.W_n = [np.random.normal(loc=0, scale=np.sqrt(2.0 / (l + next_l)), size=(l, next_l)) * np.sqrt(2. / l)
                        for l, next_l in zip(structure[:-1], structure[1:])]  # Glorot normal initialization.
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.cashZ = []
        self.cashA = []

    def initialize_Adam_optimizer(self):
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-8
        self.t = 0
        self.m_dw = [np.zeros_like(w_l) for w_l in self.W_n]
        self.v_dw = [np.zeros_like(w_l) for w_l in self.W_n]
        self.m_db = [np.zeros_like(b_l) for b_l in self.B_n]
        self.v_db = [np.zeros_like(b_l) for b_l in self.B_n]

    def initialize_Momentum_optimizer(self):
        self.beta = 0.9
        self.t_dw = [np.zeros(w_l.shape) for w_l in self.W_n]
        self.t_db = [np.zeros(b_l.shape) for b_l in self.B_n]

    def optimize(self):
        if self.optimizer == "SGD":
            self.W_n = [W - self.learning_rate * dJ_dWl for W, dJ_dWl in zip(self.W_n, self.dJ_dW)]
            self.B_n = [b - self.learning_rate * dJ_dBl for b, dJ_dBl in zip(self.B_n, self.dJ_dB)]

        elif self.optimizer == "Momentum":

            self.t_dw = [(self.beta * t_dwl) + ((1 - self.beta) * dJ_dWl) for t_dwl, dJ_dWl in
                         zip(self.t_dw, self.dJ_dW)]
            self.t_db = [(self.beta * t_dbl) + ((1 - self.beta) * dJ_dBl) for t_dbl, dJ_dBl in
                         zip(self.t_db, self.dJ_dB)]

            self.W_n = [W - self.learning_rate * dJ_dWl for W, dJ_dWl in zip(self.W_n, self.t_dw)]
            self.B_n = [b - self.learning_rate * dJ_dBl for b, dJ_dBl in zip(self.B_n, self.t_db)]

        elif self.optimizer == "Adam":
            self.t += 1

            self.m_dw = [self.beta_1 * Acfmw + ((1 - self.beta_1) * wgrad) for Acfmw, wgrad in
                         zip(self.m_dw, self.dJ_dW)]
            self.m_db = [self.beta_1 * Acfmb + ((1 - self.beta_1) * bgrad) for Acfmb, bgrad in
                         zip(self.m_db, self.dJ_dB)]

            self.v_dw = [self.beta_2 * Acsmw + ((1 - self.beta_2) * np.square(wgrad)) for Acsmw, wgrad in
                         zip(self.v_dw, self.dJ_dW)]
            self.v_db = [self.beta_2 * Acsmb + ((1 - self.beta_2) * np.square(bgrad)) for Acsmb, bgrad in
                         zip(self.v_db, self.dJ_dB)]

            # bias correction - this step could be removed if bias correction is not needed
            m_dw_hat = [corr_m / (1 - (self.beta_1 ** self.t)) for corr_m in self.m_dw]
            m_db_hat = [corr_m / (1 - (self.beta_1 ** self.t)) for corr_m in self.m_db]

            v_dw_hat = [corr_v / (1 - (self.beta_2 ** self.t)) for corr_v in self.v_dw]
            v_db_hat = [corr_v / (1 - (self.beta_2 ** self.t)) for corr_v in self.v_db]

            self.W_n = [w_l - (self.learning_rate * (m_l / (np.sqrt(v_l) + self.epsilon))) for w_l, m_l, v_l in
                        zip(self.W_n, m_dw_hat, v_dw_hat)]
            self.B_n = [b_l - (self.learning_rate * (m_l / (np.sqrt(v_l) + self.epsilon))) for b_l, m_l, v_l in
                        zip(self.B_n, m_db_hat, v_db_hat)]

    def forward(self, x, y):

        H = self.num_layers - 2  # index of last layer where the numbering starts from the first hidden layer
        self.cashZ = []
        self.cashA = []

        # forward step
        L = 0
        for b, W in zip(self.B_n, self.W_n):  # Forward pass
            z = W.T @ a + b if self.cashZ else W.T @ x + b  # if Z_n means Z_n is not empty and @ operator calls __matmul__ method
            a = self.Actfun("ReLU", z) if L != H else self.Actfun("softmax", z)
            self.cashZ.append(z)
            self.cashA.append(a)
            L += 1
        y = y.T
        loss_i = self.LossFun("CE", self.cashA[-1], y)
        y_i = np.argmax(self.cashA[-1], axis=0)

        return (np.squeeze(loss_i), y_i)

    def back_propagate(self, x, y, weight):
        """
          This function performs a backword pass for the input x

          x : An individual training example
          y: The respective class of x


          info about
          (*) represent element wise multiplication while (@) represent inner product

        """
        H = self.num_layers - 2  # index of last layer where the numbering starts from the first hidden layer
        dJ_dB = [np.zeros(b.shape) for b in self.B_n]
        dJ_dW = [np.zeros(W.shape) for W in self.W_n]

        for l in range(H, -1, -1):  # we are subtracting 1 each iteration and stop when we reach -1
            delta = self.Actfun("dReLU", self.cashZ[l]) * (self.W_n[l + 1] @ delta) if l != H else self.LossFun("dCE",
                                                                                                                y,
                                                                                                                self.cashA[
                                                                                                                    l], )
            dJ_dB[l] = delta
            dJ_dW[l] = self.cashA[l - 1] @ delta.T if l != 0 else x @ delta.T

        dJ_dB = [weight * dJ_dBl for dJ_dBl in dJ_dB]
        dJ_dW = [weight * dJ_dWl for dJ_dWl in dJ_dW]

        return (dJ_dB, dJ_dW)

    def gradient_descent(self, mini_batch, samples_weights):
        # this function perform a mini-batch gradient descent.
        self.dJ_dB = [np.zeros(b.shape) for b in self.B_n]  # list of bias vectors of all layers
        self.dJ_dW = [np.zeros(W.shape) for W in self.W_n]  # list of weight matrixes of all layers
        b_loss = 0
        b_y_pred = []

        i = 0
        for x, y in mini_batch:  # this loop goes over each example in the mini_batch
            x = np.reshape(x, (-1, 1))  # (784,1)
            y = np.reshape(y, (-1, 1))  # (10,1)
            i_loss, i_y_pred = self.forward(x, y)
            dJ_dBi, dJ_dWi = self.back_propagate(x, y, samples_weights[
                i])  # compute the gradeints of each example w.r.t networks parameters b and W
            b_loss = b_loss + i_loss
            b_y_pred.append(i_y_pred)

            self.batch_indiv_loss.append(i_loss)
            self.dJ_dB = [dJ_dBs + dJ_dBl for dJ_dBs, dJ_dBl in zip(self.dJ_dB,
                                                                    dJ_dBi)]  # accumelate the gradeints of training examoles w.r.t bias vectors in self.dJ_dB
            self.dJ_dW = [dJ_dWs + dJ_dWl for dJ_dWs, dJ_dWl in zip(self.dJ_dW,
                                                                    dJ_dWi)]  # accumelate the gradeints of training examoles w.r.t weight matrixes in self.dJ_dW
            i += 1

        # take one gradeint step over a mini-batch

        self.optimize()
        return ((b_loss / self.batch_size), b_y_pred)

    def train(self, epochs, x_train, y_train, T_indxs=None, N_indxs=None, alpha=0.1, weighting='No'):

        self.epochs = epochs
        self.N = x_train.shape[0]
        self.samples_weights = np.ones(shape=(self.N, 1), dtype=np.float32)
        self.weighting = weighting

        generator = BatchGenerator(x_col=x_train, y_col=y_train, batch_size=self.batch_size, shuffle=False)
        n_batches = len(generator)
        batch_loss = np.zeros((n_batches, 1), dtype=np.float32())
        train_indiv_losses = np.zeros(shape=(self.epochs, self.N), dtype=np.float32)

        round = 0
        counter = np.zeros(shape=(self.N, 1), dtype=np.float32)

        if self.optimizer == "Adam":
            self.initialize_Adam_optimizer()
        elif self.optimizer == "Momentum":
            self.initialize_Momentum_optimizer()

        if weighting == 'No':
            pass
        else:
            ws_c = WeightingStrategies(alpha, self.N)
            print("using weighting {}".format(weighting))

        # create mini_batches
        for epoch in range(epochs):
            batches_preds = []
            for i in range(n_batches):
                self.batch_indiv_loss = []
                x, y, indexes = generator[i]
                batch = list(zip(x, y))
                batch_sw = self.samples_weights[indexes] / np.sum(self.samples_weights[indexes])
                batch_loss[i], preds = self.gradient_descent(batch, batch_sw)  # wieghted loss function
                batches_preds.append(preds)
                train_indiv_losses[epoch, indexes] = np.array(self.batch_indiv_loss)

            # computing sample weights
            if (epoch != 0) and ((epoch + 1) % 10) == 0 and self.weighting in ["one", "two", "three", "four"]:
                if weighting == 'four':
                    self.samples_weights = ws_c("generate_sample_weights_4", train_indiv_losses[epoch - 9:epoch + 1, :])
                elif weighting == 'one':
                    self.samples_weights = ws_c("generate_sample_weights_1", train_indiv_losses[epoch - 9:epoch + 1, :])
                elif weighting == 'two':
                    self.samples_weights = ws_c("generate_sample_weights_2", train_indiv_losses[epoch - 9:epoch + 1, :])
                elif weighting == 'three':
                    round += 1
                    self.samples_weights, counter = ws_c("generate_sample_weights_3",
                                                         train_indiv_losses[epoch - 9:epoch + 1, :], round, counter)

                print("sum of noisy weights: {:.11f}".format(np.average(self.samples_weights[N_indxs])))
                print("sum of typical weights: {:.11f}".format(np.average(self.samples_weights[T_indxs])))
            else:
                pass
            epoch_preds = [preds for batch in batches_preds for preds in batch]
            accuracy = np.mean(np.reshape(np.argmax(y_train, axis=1), (-1, 1)) == np.array(epoch_preds))
            print("Epoch {}/{}, Training loss: {}, Training accuracy: {}".format(epoch, epochs, np.mean(batch_loss),
                                                                                 accuracy))

    def predict(self, x, y):
        """
          evaluate the nueral model on test data

          x: test data features
          y: test data labels

        """
        acc_rate = []
        loss = np.zeros((y.shape[0],))

        for i in range(x.shape[0]):
            X_i = np.reshape(x[i, :], (-1, 1))
            Y_i = np.reshape(y[i, :], (-1, 1))
            loss_i, y_hat_i = self.forward(X_i, Y_i)
            loss[i] = loss_i
            acc_rate.append(np.argmax(Y_i, axis=0) == y_hat_i)

        # accumulate the losses and accuracy of training data in loss_test and acc_test,respectively
        error_rate = np.mean(loss_i)
        accuracy = np.mean(np.array(acc_rate))

        # test accuracy and error rate

        print("Test error rate: {}, test accuracy: {}".format(error_rate, accuracy))


# ---------------------------------------------------------------------------------
def generating_dataset(noise_rate=0):
    # load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    image_vector_size = x_train.shape[1] * x_train.shape[1]
    num_classes = len(np.unique(y_train))

    # Flatten the images
    x_train = x_train.reshape(x_train.shape[0], image_vector_size)
    x_test = x_test.reshape(x_test.shape[0], image_vector_size)

    # Asymmetrical flip
    if noise_rate != 0:
        np.random.seed(2)
        random.seed(2)
        seed(2)
        y_noise = np.copy(y_train)
        c = range(num_classes)
        noisy_indexes = []
        typical_indexes = []

        for i in c:
            y_i = np.where(y_train == i)[0].flatten()
            N = math.floor(noise_rate * y_i.size)
            p = np.random.choice(y_i, N, replace=False).flatten()
            noisy_indexes.extend(p)
            if (i != (c[-1])):
                y_noise[p] = i + 1
            else:
                y_noise[p] = c[0]

        typical_indexes = np.delete(np.array(range(y_noise.size)), np.array(noisy_indexes))
        y_train = y_noise
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)  # (10000,10)

        return (x_train, y_train), (x_test, y_test), num_classes, typical_indexes, noisy_indexes
    else:
        # Convert to "one-hot" vectors using the to_categorical function
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)  # (10000,10)
        return (x_train, y_train), (x_test, y_test), num_classes


def main():
    np.random.seed(2)
    random.seed(2)
    seed(2)
    ActFun = ActivationFuncs()
    LossFun = LossFuncs()
    epochs = 200
    batch_size = 200
    optimizer = "Adam"
    lr = 0.001
    noise_rate = 0.4
    weighting = "No"

    # Import dataset
    if (noise_rate != 0):
        (x_train, y_train), (x_test, y_test), num_classes, T_indxs, N_indxs = generating_dataset(noise_rate=noise_rate)
    else:
        (x_train, y_train), (x_test, y_test), num_classes = generating_dataset()

    # Create NN with 3 hidden layers with each of size 30
    Net_Structure = [x_train.shape[1], 30, 30, 30, num_classes]
    my_net = Network(Net_Structure, ActFun, LossFun, optimizer, lr, batch_size, initialization='Glorot_Normal')

    # TRAIN AND TEST THE NETWORK ON X_TRAIN AND X_TEST
    if (noise_rate != 0):
        my_net.train(epochs, x_train, y_train, T_indxs, N_indxs, weighting=weighting)
    else:
        my_net.train(epochs, x_train, y_train)

    my_net.predict(x_test, y_test)


# Entry point
if __name__ == '__main__':
    main()
