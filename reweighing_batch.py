# insert the directory ColabNotebooks to my python path using sys:


import numpy as np
import random
from numpy.random import seed
import math
from batchgenerator import *
from WeightingStrategies import *
from NNFunctions import *
import tensorflow as tf
from keras.datasets import mnist


class Network:
    def __init__(self, structure, Actfun, LossFun, optimizer, learning_rate, batch_size,
                 initialization='He initialization'):
        self.structure = structure
        self.num_layers = len(structure)
        self.Actfun = Actfun
        self.LossFun = LossFun
        self.B_n = [np.zeros((1, l)) for l in structure[1:]]
        if initialization == 'He initialization':
            self.W_n = [np.random.randn(l, next_l) * np.sqrt(2. / l) for l, next_l in
                        zip(structure[:-1], structure[1:])]  # He initialization "
        elif initialization == 'Glorot Normal':
            self.W_n = [np.random.normal(loc=0, scale=np.sqrt(2.0 / (l + next_l)), size=(l, next_l)) * np.sqrt(2. / l)
                        for l, next_l in zip(structure[:-1], structure[1:])]  # Glorot normal initialization.
        elif initialization == 'Glorot Uniform':
            self.W_n = [np.random.uniform(low=-np.sqrt(6.0 / (l + next_l)), high=np.sqrt(6.0 / l + next_l),
                                          size=(l, next_l)) * np.sqrt(2. / l)
                        for l, next_l in zip(structure[:-1], structure[1:])]  # Glorot uniform initialization
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.cashZ = []
        self.cashA = []

        self.dJ_dB = [np.zeros(b.shape) for b in self.B_n]  # list of bias vectors of all layers
        self.dJ_dW = [np.zeros(W.shape) for W in self.W_n]  # list of weight matrixes of all layers

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

    def batch_forward(self, x_batch, y_batch):

        H = self.num_layers - 2  # index of last layer where the numbering starts from the first hidden layer
        self.cashZ = []
        self.cashA = []

        # forward step
        L = 0
        for b, W in zip(self.B_n, self.W_n):  # Forward pass
            z = np.dot(a, W) + b if self.cashZ else np.dot(x_batch,
                                                           W) + b  # if Z_n means Z_n is not empty and @ operator calls __matmul__ method
            a = self.Actfun("ReLU", z) if L != H else self.Actfun("softmax_b", z)
            self.cashZ.append(z)
            self.cashA.append(a)
            L += 1

        A_l, self.batch_indiv_loss = self.LossFun("Batch_CE", y_batch, self.cashA[-1])
        y_batch_preds = np.argmax(self.cashA[-1], axis=1)

        return np.squeeze(A_l), y_batch_preds

    def batch_backward(self, x, y, batch_sw):

        """
        This function performs a backward pass for the input x

        x : An individual training example
        y: The respective class of x


        info about
        (*) represent element wise multiplication while (@) represent inner product

        """
        H = self.num_layers - 2  # index of last layer where the numbering starts from the first hidden layer
        # backward step, the sample weights must has shape (1,batch_size)
        for l in range(H, -1, -1):  # we are subtracting 1 each iteration and stop when we reach -1
            delta = np.dot(delta, self.W_n[l + 1].T) * self.Actfun("dReLU", self.cashZ[l]) if l != H else self.LossFun(
                "dCE", y, self.cashA[l])
            self.dJ_dB[l] = np.dot(batch_sw.T,
                                   delta)  # np.sum will do the following : for each coloumn it will sum the elements for all the rwos. hense the shape of np.sum is (10,1)
            self.dJ_dW[l] = np.dot(self.cashA[l - 1].T, delta * batch_sw) if l != 0 else np.dot(x.T, delta * batch_sw)

    def train(self, epochs, x_train, y_train, T_indxs=None, N_indxs=None, weighting='No', alpha=0.1):

        self.epochs = epochs
        self.N = x_train.shape[0]
        self.samples_weights = np.ones(shape=(self.N, 1), dtype=np.float32)
        self.weighting = weighting

        # create mini_batches
        generator = BatchGenerator(x_col=x_train, y_col=y_train, batch_size=self.batch_size, shuffle=False)
        n_batches = len(generator)
        batch_loss = np.zeros((n_batches, 1), dtype=np.float32())
        train_indiv_losses = np.zeros(shape=(self.epochs, self.N), dtype=np.float32)
        EMA_L = None
        round = 0
        counter = np.zeros(shape=(self.N, 1), dtype=np.float32)

        if self.optimizer == "Adam":
            self.initialize_Adam_optimizer()
        elif self.optimizer == "Momentum":
            self.initialize_Momentum_optimizer()

        if self.weighting == "No":
            pass
        else:
            ws_c = WeightingStrategies(alpha, self.N)
            print("using weighting {} ".format(weighting))

        for epoch in range(self.epochs):
            batches_preds = []
            for i in range(n_batches):
                self.batch_indiv_loss = []
                x, y, indexes = generator[i]
                batch_sw = self.samples_weights[indexes] / np.sum(self.samples_weights[indexes])
                batch_loss[i], predcs = self.batch_forward(x, y)
                self.batch_backward(x=x, y=y, batch_sw=batch_sw)
                self.optimize()
                batches_preds.append(predcs)
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

                EMA_L = np.zeros(shape=(self.N, 1), dtype=np.float32)

                print("sum of noisy weights: {:.11f}".format(np.average(self.samples_weights[N_indxs])))
                print("sum of typical weights: {:.11f}".format(np.average(self.samples_weights[T_indxs])))
            else:
                pass

            epoch_preds = [preds for batch in batches_preds for preds in batch]
            accuracy = np.mean(np.argmax(y_train, axis=1) == np.array(epoch_preds))
            print("Epoch {}/{}, Training loss: {}, Training accuracy: {} %".format(epoch, self.epochs,
                                                                                   np.mean(batch_loss),
                                                                                   (accuracy * 100)))

    def predict(self, x, y):
        """
        evaluate the neural model on test data
  
        x: test data features
        y: test data labels

        """

        y_hat = np.zeros((y.shape[0],))
        loss = np.zeros((y.shape[0],))
        loss, y_hat = self.batch_forward(x, y)
        accuracy = np.mean(np.array(np.argmax(y, axis=1) == y_hat))

        print("Test error rate: {}, test accuracy: {} %".format(loss, (accuracy * 100)))


# ---------------------------------------------------------------------------------
def generating_dataset(noise_rate=0):
    # load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    image_vector_size = x_train.shape[1] * x_train.shape[1]
    num_classes = len(np.unique(y_train))

    # Flatten the images
    x_train = x_train.reshape(x_train.shape[0], image_vector_size)
    x_test = x_test.reshape(x_test.shape[0], image_vector_size)

    # Normalize the images
    # x_train, x_test = x_train / 255.0, x_test / 255.0

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
            if i != (c[-1]):
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


# ---------------------------------------------------------------------------------

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
    weighting = "four"

    # Import dataset
    if noise_rate != 0:
        (x_train, y_train), (x_test, y_test), num_classes, T_indxs, N_indxs = generating_dataset(noise_rate=noise_rate)
    else:
        (x_train, y_train), (x_test, y_test), num_classes = generating_dataset()

    # Create NN with 3 hidden layers with each of size 30
    Net_Structure = [x_train.shape[1], 30, 30, 30, num_classes]
    my_net = Network(Net_Structure, ActFun, LossFun, optimizer, lr, batch_size, initialization='Glorot Normal')

    # TRAIN AND TEST THE NETWORK ON X_TRAIN AND X_TEST
    if noise_rate != 0:
        my_net.train(epochs, x_train, y_train, T_indxs=T_indxs, N_indxs=N_indxs, weighting=weighting)
    else:
        my_net.train(epochs, x_train, y_train)

    my_net.predict(x_test, y_test)


# batch_training

# Entry point
if __name__ == '__main__':
    main()
