import numpy as np
from sklearn.metrics import f1_score


# Metrics
# y_: output y: gt
# if largest number index is not 1, then is error
class metric(object):
    def __init__(self, f):
        self.f = f

    def __call__(self, y_, y, *args, **kwargs):
        y_ = y_.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        metric = self.f(y_, y, *args, **kwargs)
        return metric


@metric
def f1(y_, y):
    y_[y_ >= threshold] = 1
    y_[y_ < threshold] = 0
    y[y >= threshold] = 1
    y[y < threshold] = 0

    y_binary = np.zeros(y_.shape)

    for batch in range(y_.shape[0]):
        y_binary[batch, np.argmax(y_[batch, :, :], axis=0),
                 np.arange(y_.shape[-1])] = 1

    y_binary = y_binary.transpose(1, 0, 2).reshape(y_binary.shape[1], -1)
    y = y.transpose(1, 0, 2).reshape(y.shape[1], -1)

    return f1_score(y, y_binary, average='micro')  # f1_batch / y.shape[0]


@metric
def accuracy(y_, y):
    y_binary = np.zeros(y_.shape)

    for batch in range(y_.shape[0]):
        y_binary[batch, np.argmax(y_[batch, :, :], axis=0),
                 np.arange(y_.shape[-1])] = 1

    y_mul = y_binary * y
    y_mul_sum = np.sum(y_mul, axis=1)

    acc = np.sum(y_mul_sum) / y_mul_sum.size

    return acc
