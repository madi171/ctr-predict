'''
Author AlexMa
Date: 2015-11-21
'''

import random
import math
import sys

from datetime import datetime
from math import exp, log, sqrt
from sklearn import metrics


##############################################################################
# parameters #################################################################
##############################################################################


train = 'datasets/train.shuf'               # path to training file
test = 'datasets/test.shuf'                 # path to testing file

# about model
alpha = 0.1     # learning rate
beta = 1.       # smoothing parameter for adaptive learning rate
L1 = 1.         # L1 regularization, larger value means more regularized
L2 = 1.         # L2 regularization, larger value means more regularized

# about feature/hash trick
D = 160000               # number of weights to use
do_interactions = True  # whether to enable poly2 feature interactions
if do_interactions:
    D = 160000 + 100000000

# about training/validation
epoch = 1      # learn training data for N passes
holdout = 100  # use every N training instance for holdout validation

# other parameters
threshold = 0.5 # separate positive and negative classes by threshold
sampling_rate = 0.2 # take this as a precentage of randomly selection on negative class
class_positive_boost = 2.0 # boosting factor on positive class


##############################################################################
# class, function, generator definitions #####################################
##############################################################################

# implemented follow the google ad-click-predict paper
class ftrl_proximal(object):
    ''' Our main algorithm: Follow the regularized leader - proximal

        In short,
        this is an adaptive-learning-rate sparse logistic-regression with
        efficient L1-L2-regularization

        Reference:
        http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
    '''

    def __init__(self, alpha, beta, L1, L2, D, interaction=False):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2

        # feature related parameters
        self.D = D
        self.interaction = interaction

        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        self.n = [0.] * D
        self.z = [0.] * D
        self.w = [0.] * D  # use this for execution speed up
        # self.w = {}  # use this for memory usage reduction

    def _indices(self, x):
        ''' A helper generator that yields the indices in x

            The purpose of this generator is to make the following
            code a bit cleaner when doing feature interaction.

            x is a key-value formmat feature vector
        '''

        for i in x.keys():
            yield i

        #if self.interaction:
        #    D = self.D
        #    L = len(x)
        #    for i in xrange(, L):  # skip bias term, so we start at 1
        #        for j in xrange(i+1, L):
        #            yield (i * j) % D

    def predict(self, x):
        ''' Get probability estimation on x

            INPUT:
                x: features

            OUTPUT:
                probability of p(y = 1 | x; w)
        '''

        # parameters
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        # model
        n = self.n
        z = self.z
        w = self.w  # use this for execution speed up
        # w = {}  # use this for memory usage reduction

        # wTx is the inner product of w and x
        wTx = 0.
        for i in self._indices(x):
            sign = -1. if z[i] < 0 else 1.  # get sign of z[i]

            # build w on the fly using z and n, hence the name - lazy weights -
            if sign * z[i] <= L1:
                # w[i] vanishes due to L1 regularization
                w[i] = 0.
            else:
                # apply prediction time L1, L2 regularization to z and get w
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)

            # calculate wTx inner production
            wTx += w[i] * x[i]

        self.w = w

        # bounded sigmoid function, this is the probability estimation
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, p, y):
        ''' Update model using x, p, y

            INPUT:
                x: feature, a list of indices and weights
                p: click probability prediction of our model
                y: answer

            MODIFIES:
                self.n: increase by squared gradient
                self.z: weights
        '''

        # parameter
        alpha = self.alpha

        # model
        n = self.n
        z = self.z
        w = self.w  # no need to change this, it won't gain anything

        # gradient under logloss
        g = p - y

        # update z and n
        for i in self._indices(x):
            g = (p - y) * x[i]
            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
            z[i] += g - sigma * w[i]
            n[i] += g * g


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
                _id = int(fs[0])
                _w = float(fs[1])
                if y > 0.5:
                    # positive boosting
                    x[_id] = _w * class_positive_boost
                else:
                    # negative not boosting
                    x[_id] = _w

            if do_interactions:
                # do topic feature interaction,from 2000 - 8000
                key_set = [k if (k >= 2000 and k <= 8000) else None for k in x.keys()]
                for i in key_set:
                    for j in key_set:
                        if i >= j or i is None or j is None:
                            continue
                        x[i * 1000 + j] = x[i] * x[j]
            yield counter, x, y
        print >> sys.stderr, "Using interaction=%d, self.D=%d" % (do_interactions, D)

def dump_weight(model):
    ''' FUNCTION: dump weights matrix to dict

        INPUT:
            model: our weights matrix

        OUTPUT:
            dict
    '''
    weight = {}
    for i in xrange(0, len(model.w)):
        if model.w[i] > 0.00001:
            weight[i] = model.w[i]
    ret = sorted(weight.items(), key=lambda x:x[1], reverse=True)
    print ret[:200]


##############################################################################
# start training #############################################################
##############################################################################

start = datetime.now()
if len(sys.argv) == 6:
    alpha = float(sys.argv[1])
    beta = float(sys.argv[2])
    sampling_rate = float(sys.argv[3])
    L1 = float(sys.argv[4])
    L2 = float(sys.argv[5])

# initialize ourselves a learner
learner = ftrl_proximal(alpha, beta, L1, L2, D, interaction=do_interactions)
print "Start ftrl with param{alpha:%f, beta:%f, samplingrate:%f, l1:%f, l2:%f}" % (alpha, beta, sampling_rate, L1, L2)

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
for t, x, y in data(test, D, True):  # data is a generator
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

print "total auc=%.6f with param{alpha:%f, beta:%f, samplingrate:%f, l1:%f, l2:%f}" % (metrics.roc_auc_score(y_true, y_pred), alpha, beta, sampling_rate, L1, L2)
print "Dump weight:"
for w in dump_weight(learner):
    print w
