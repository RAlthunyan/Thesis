import math
import random

import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Dense, Input
from keras.models import Model
from numpy.random import seed
from tensorflow.keras.optimizers import Adam

beta = 0.1
epochs = 200
batch_size = 200
N = 60000
learning_rate = 0.001
num_classes = 0
image_vector_size = 0


class batchgenerator(tf.keras.utils.Sequence):

    def __init__(self, x_col, y_col, batch_size=100, shuffle=False):
        self.x_col = x_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.key_array = np.arange(self.x_col.shape[0])
        self.shuffle = shuffle

    def __len__(self):
        return len(self.key_array) // self.batch_size  # return number of batches

    def __getitem__(self, index):
        keys = self.key_array[index * self.batch_size: (index + 1) * self.batch_size]
        x = np.asarray(self.x_col[keys])
        y = np.asarray(self.y_col[keys])

        return x, y, keys

    def on_epoch_end(self):
        if self.shuffle:
            self.key_array = np.random.permutation(self.key_array)


def keras_model():
    inputs = Input(shape=(image_vector_size,))
    x = Dense(32, activation='relu')(inputs)
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=inputs, outputs=outputs, name='MNIST_MODEL')


def generating_dataset():
    # load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    global image_vector_size
    image_vector_size = x_train.shape[1] * x_train.shape[1]
    global num_classes
    num_classes = len(np.unique(y_train))

    # Flatten the imges
    x_train = x_train.reshape(x_train.shape[0], image_vector_size)
    x_test = x_test.reshape(x_test.shape[0], image_vector_size)

    # Normalize the images
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Convert to "one-hot" vectors using the to_categorical function
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)  # (10000,10)

    return (x_train, y_train), (x_test, y_test)


def ECDF(data):
    x = np.sort(data)
    n = x.shape[0]
    y = np.arange(1, n + 1) / n
    return y


def w_loss(sample_weights, y_true, y_pred):
    wtd_loss = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)(y_true, y_pred,
                                                                                                sample_weights)
    avg_loss = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)(y_true,
                                                                                                                y_pred)
    indiv_losses = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred)

    return wtd_loss, avg_loss, indiv_losses


def generate_sample_weights_1(EMA_Loss):
    # Scaling Data Between 0 and 1 using min-max normalization
    EMA_Loss_scaled = (EMA_Loss - np.min(EMA_Loss)) / (np.max(EMA_Loss) - np.min(EMA_Loss))

    # compute the weights of all training data
    scores = 1 - EMA_Loss_scaled

    return scores


# second weighting strategy

def generate_sample_weights_2(EMA_Loss):
    scores = np.zeros(shape=(N, 1), dtype=np.float32)

    # Scalling Data Between 0 and 1 using min-max normalization
    EMA_Loss_scaled = (EMA_Loss - np.min(EMA_Loss)) / (np.max(EMA_Loss) - np.min(EMA_Loss))

    # compute the upper bound
    # mean = np.average(EMA_Loss_scaled)
    # SD = np.std(EMA_Loss_scaled)
    # u_upper = (0.5*SD)+mean

    # adjusted approach
    # for i , value in enumerate(EMA_Loss_scaled):
    #   if (value<= u_upper):
    #     scores[i] = (1/batch_size) * np.exp((value*round))
    #   else:
    #     scores[i] = (1/batch_size) * np.exp(-(value*round))

    # compute the weights of all training data
    scores = np.exp(-EMA_Loss_scaled)

    return scores


# Third weighting strategy

def generate_sample_weights_3(EMA_Loss, r, counter):
    # Scaling Data Between 0 and 1 using min-max normalization
    EMA_Loss_scaled = (EMA_Loss - np.min(EMA_Loss)) / (np.max(EMA_Loss) - np.min(EMA_Loss))
    mean = np.average(EMA_Loss_scaled)
    SD = np.std(EMA_Loss_scaled)
    T = (2 * SD) + mean

    for i, l in np.ndenumerate(EMA_Loss_scaled):
        if l <= T:
            counter[i] += 1

    input_tupels = list(zip(EMA_Loss_scaled, counter))  # create list of tupels
    # compute the weights of training data

    scores = np.asarray([(1 - loss) * (c / r) for i, (loss, c) in enumerate(input_tupels)])
    return scores, counter


def computing_EMA(t_loss):
    EMA = []
    rows = 10
    for j in range(N):
        t = []
        for i in range(rows):
            if i == 0:
                t.append(t_loss[i][j])
            else:
                e = ((1 - beta) * t[-1]) + (beta * t_loss[i][j])
                t.append(e)

        EMA.extend(t)

    EMA_L = np.array(EMA).reshape(N, rows)
    return EMA_L[:, -1]


# fourth weighting strategy
def generate_sample_weights_4(EMA_Loss):
    # compute the upper loss values of clean data
    mean = np.average(EMA_Loss)
    SD = np.std(EMA_Loss)
    u_upper = (2 * SD) + mean

    # compute the sample weights
    scores = (EMA_Loss - np.min(EMA_Loss)) / (u_upper - np.min(EMA_Loss))
    scores = np.exp(-scores)

    return scores


def asymmetry_flip(y_train, noise_rate):
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
        if (i != c[-1]):
            y_noise[p] = i + 1
        else:
            y_noise[p] = c[0]

    typical_indexes = np.delete(np.array(range(y_noise.shape[0])), np.array(noisy_indexes))
    return y_noise, noisy_indexes, list(typical_indexes)


def is_index_selected(selected_indxs, typical_indxs, noisy_indxs):
    selected_T_indxs = [index for index in typical_indxs if index in selected_indxs]
    selected_n_indxs = [index for index in noisy_indxs if index in selected_indxs]

    return selected_T_indxs, selected_n_indxs


#  The following function perform the NN training
def train():
    seed(2)
    tf.random.set_seed(2)
    random.seed(2)
    (x_train, y_train), (x_test, y_test) = generating_dataset()

    # Generating noisy data by flipping labels in a nonrandom way
    y_noisy, noisy_labels, typical_labels = asymmetry_flip(y_train, 0.4)
    y_noisy = tf.keras.utils.to_categorical(y_noisy, num_classes)

    # ---------------------------------------------------------------------------------#
    MNIST_model = keras_model()
    MNIST_model.summary()
    optimizer = Adam(learning_rate)  # identifying the optimizer

    generator = batchgenerator(x_col=x_train, y_col=y_noisy, batch_size=batch_size, shuffle=False)
    n_batches = len(generator)

    loss_train = np.zeros(shape=(epochs,), dtype=np.float32)  # 1D array to store the training set loss over epochs
    acc_train = np.zeros(shape=(epochs,), dtype=np.float32)
    epoch_loss_indiv = np.zeros(shape=(epochs, N),
                                dtype=np.float32)  # bookkeeping variable to track the loss of each individual instances over epochs
    samples_weights = np.ones(shape=(N,), dtype=np.float32)
    epoch_loss_avg = tf.keras.metrics.Mean()  # It is object to average all bathces losses
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    round = 0
    counter = np.zeros(shape=(N,), dtype=np.float32)

    # ----------------------------------------------------------------------------------#

    for epoch in range(epochs):
        for batch in range(n_batches):
            x, y, indexes = generator[batch]
            batch_sw = samples_weights[indexes] / np.sum(samples_weights[indexes])

            with tf.GradientTape() as tape:  # Forward pass. tf.GradientTape is a tool recording all operations that take place under it
                y_ = MNIST_model(x, training=True)
                wts_loss, avg_loss, loss_indv = w_loss(batch_sw.T, y_true=y, y_pred=y_)

            grad = tape.gradient(wts_loss, MNIST_model.trainable_variables)  # backpropagation
            optimizer.apply_gradients(zip(grad,
                                          MNIST_model.trainable_variables))  # zib is to map the gradient to the trainable parameters in order to update network weights
            epoch_loss_indiv[epoch, indexes] = loss_indv
            epoch_loss_avg.update_state(avg_loss)
            train_acc_metric.update_state(y, y_)

        loss_train[
            epoch] = epoch_loss_avg.result()  # After accumulating the loss over batches in epoch_loss_avg object, we called result to average these losses to get the average loss of one epoch
        acc_train[epoch] = train_acc_metric.result()
        print("Epoch {}/{}, Training loss: {}, Training Accuracy: {}".format(epoch, epochs, loss_train[epoch],
                                                                             acc_train[epoch] * 100))
        epoch_loss_avg.reset_states()
        train_acc_metric.reset_states()

        if (epoch != 0) and ((epoch + 1) % 10) == 0:
            round += 1
            EMA_t = computing_EMA(epoch_loss_indiv[epoch - 9:epoch + 1, :])
            # Compute the weights of training data based on the EMA of losses
            samples_weights = generate_sample_weights_4(EMA_t)
            # printing average weights of noisy and non-noisy examples
            print("average of noisy weights: {:.11f}".format(np.average(samples_weights[noisy_labels])))
            print("average of typical weights: {:.11f}".format(np.average(samples_weights[typical_labels])))
        else:
            pass

    # In order to calculate test accuracy using evaluate function we first need to compile the MNIST_model
    MNIST_model.compile(optimizer=optimizer, loss=tf.keras.losses.categorical_crossentropy, metrics=["acc"])

    (loss, acc) = MNIST_model.evaluate(x_test, y_test)
    print("[INFO] test accuracy: {:.4f}".format(acc * 100))


train()
