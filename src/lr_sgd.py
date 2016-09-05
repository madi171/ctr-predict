'''
Author: AlexMa
Date: 2016-01-25
'''


from datetime import datetime
from math import exp, log, sqrt
from sklearn import metrics
import random
import math


##############################################################################
# parameters #################################################################
##############################################################################

train = 'datasets/train.shuf'               # path to training file
test = 'datasets/test.shuf'                 # path to testing file

# B, model
alpha = 0.005  # learning rate
#L1 = 1.     # L1 regularization, larger value means more regularized
#L2 = 1.     # L2 regularization, larger value means more regularized

# feature/hash trick
D = 160000               # number of weights to use
do_interactions = False  # whether to enable poly2 feature interactions

# training/validation
epoch = 1      # learn training data for N passes
holdout = 100  # use every N training instance for holdout validation

threshold = 0.5
sampling_rate = 0.2

class sgd_optimizate(object):
    def __init__(self, alpha, D):
        # parameters
        self.alpha = alpha

        # feature related parameters
        self.D = D

        # model
        # w: lazy weights
        self.w = [0.] * D  # use this for execution speed up

    def _indices(self, x):
        ''' A helper generator that yields the indices in x

            The purpose of this generator is to make the following
            code a bit cleaner when doing feature interaction.

            x is a key-value formmat feature vector
        '''

        for i in x.keys():
            yield i

    def predict(self, x):
        ''' Get probability estimation on x

            INPUT:
                x: features

            OUTPUT:
                probability of p(y = 1 | x; w)
        '''

        # parameters
        alpha = self.alpha

        # model
        w = self.w  # use this for execution speed up

        # wTx is the inner product of w and x
        wTx = 0.
        for i in self._indices(x):
            wTx += w[i] * x[i]

        # bounded sigmoid function, this is the probability estimation
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, p, y):
        ''' Update model using x, p, y

            INPUT:
                x: feature, a list of indices and weights
                p: click probability prediction of our model
                y: answer

            MODIFIES:
                self.w: weights
        '''

        # parameter
        alpha = self.alpha

        # gradient under logloss
        g = p - y

        # update w
        for i in self._indices(x):
            g = (p - y) * x[i]
            self.w[i] = self.w[i] - self.alpha * g


def logloss(p, y):
    ''' FUNCTION: Bounded logloss

        INPUT:
            p: our prediction
            y: real answer

        OUTPUT:
            logarithmic loss of p given y
    '''

    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)


def data(path, D, sampling=False):
    ''' GENERATOR: Apply hash-trick to the original csv row
                   and for simplicity, we one-hot-encode everything

        INPUT:
            path: path to training or testing file
            D: the max index that we can hash to

        YIELDS:
            ID: id of the instance, mainly useless
            x: a list of hashed and one-hot-encoded 'indices'
               we only need the index since all values are either 0 or 1
            y: y = 1 if we have a click, else we have y = 0
    '''
    with open(path) as fobj:
        counter = 0
        for line in fobj:
            y, ts, features = line.strip().split('\t')
            counter += 1
            y = float(y)

            # do negative sampling
            if sampling:
                if y < 0.5 and random.uniform(0,1) > sampling_rate:
                    continue

            x = {}
            for f in features.split(','):
                fs = f.split(':')
                if len(fs) != 2:
                    continue
                x[int(fs[0])] = float(fs[1])
            yield counter, x, y



##############################################################################
# start training #############################################################
##############################################################################

start = datetime.now()

# initialize ourselves a learner
learner = sgd_optimizate(alpha, D)

# start training
for e in xrange(epoch):
    loss = 0.
    count = 0

    for t, x, y in data(train, D, True):  # data is a generator
        #  t: just a instance counter
        #  x: features, indices:weights
        #  y: label (click)

        # step 1, get prediction from learner
        p = learner.predict(x)

        if t % holdout == 0:
            # step 2-1, calculate holdout validation loss
            #           we do not train with the holdout data so that our
            #           validation loss is an accurate estimation of
            #           the out-of-sample error
            loss += logloss(p, y)
            count += 1
        else:
            # step 2-2, update learner with label (click) information
            learner.update(x, p, y)

        if t % 50000 == 0 and t > 1:
            print(' %s\tencountered: %d\tcurrent logloss: %f' % (
                datetime.now(), t, loss/count))

    print('Epoch %d finished, holdout logloss: %f, elapsed time: %s' % (
        e, loss/count, str(datetime.now() - start)))


##############################################################################
# start testing, and build Kaggle's submission file ##########################
##############################################################################
def acc(x,y,th):
    l = [0 if math.fabs(t[0] - t[1]) > th else 1 for t in zip(x, y)]
    return 1.0 * sum(l) / len(y)

print "---------------------Train Done----------------"
y_true = []
y_pred = []
t = 0
for t, x, y in data(test, D):  # data is a generator
    #  t: just a instance counter
    #  x: features, indices:weights
    #  y: label (click)

    # step 1, get prediction from learner
    p = learner.predict(x)
    y_true.append(y)
    y_pred.append(p)
    if t % 30000 == 0:
        print(' %s\ttest encountered: %d\tcurrent acc: %f' % (
                datetime.now(), t, acc(y_true, y_pred, threshold)))
    t += 1

print "total auc=%.6f" % (metrics.roc_auc_score(y_true, y_pred))
