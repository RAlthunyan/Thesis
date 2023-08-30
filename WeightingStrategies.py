import numpy as np


class WeightingStrategies:
    def __init__(self, alpha, N):
        self.N = N
        self.alpha = alpha

    def __call__(self, func, *args):  # to make the object as callable function redifine the same object

        result = getattr(self, func)(*args)
        return result

    def generate_sample_weights_1(self, t_Loss):

        EMA_Loss = self.computing_EMA(t_Loss)

        # Scaling Data Between 0 and 1 using min-max normalization
        EMA_Loss_scaled = (EMA_Loss - np.min(EMA_Loss)) / (np.max(EMA_Loss) - np.min(EMA_Loss))

        # compute the weights of all training data
        scores = 1 - EMA_Loss_scaled
        SW = scores.reshape(-1, 1)

        return SW

    def generate_sample_weights_2(self, t_Loss):

        EMA_Loss = self.computing_EMA(t_Loss)

        # Scaling Data Between 0 and 1 using min-max normalization
        EMA_Loss_scaled = (EMA_Loss - np.min(EMA_Loss)) / (np.max(EMA_Loss) - np.min(EMA_Loss))
        scores = np.exp(-EMA_Loss_scaled)
        SW = scores.reshape(-1, 1)

        return SW

    def generate_sample_weights_3(self, t_Loss, r, counter):

        EMA_Loss = self.computing_EMA(t_Loss)

        # Scaling Data Between 0 and 1 using min-max normalization
        EMA_Loss_scaled = (EMA_Loss - np.min(EMA_Loss)) / (np.max(EMA_Loss) - np.min(EMA_Loss))

        mean = np.average(EMA_Loss_scaled)
        SD = np.std(EMA_Loss_scaled)
        T = (2 * SD) + mean

        for i, l in np.ndenumerate(EMA_Loss_scaled):
            if l <= T:
                counter[i] += 1

        input_tupels = list(zip(EMA_Loss_scaled, counter))  # create list of tupels
        scores = np.asarray([(1 - loss) * (c / r) for i, (loss, c) in enumerate(input_tupels)])
        SW = scores.reshape(-1, 1)
        # compute the weights of training data

        return SW, counter

    def generate_sample_weights_4(self, t_Loss):
        """
        t_loss : a matrix of previous loss of 10 epochs for all training examples

        """

        EMA_Loss = self.computing_EMA(t_Loss)

        # compute the upper loss values of clean data

        mean = np.average(EMA_Loss)
        SD = np.std(EMA_Loss)
        u_upper = (0.5 * SD) + mean

        scores = (EMA_Loss - np.min(EMA_Loss)) / (u_upper - np.min(EMA_Loss))
        scores = np.exp(-(scores))
        SW = scores.reshape(-1, 1)

        # compute the weights of training data

        return SW

    def computing_EMA(self, t_loss):
        EMA = []
        rows = 10
        for j in range(self.N):
            t = []
            for i in range(rows):
                if i == 0:
                    t.append(t_loss[i][j])
                else:
                    e = ((1 - self.alpha) * t[-1]) + (self.alpha * t_loss[i][j])
                    t.append(e)

            EMA.extend(t)
        EMA_L = np.array(EMA).reshape(self.N, rows)

        return EMA_L[:, -1].T
