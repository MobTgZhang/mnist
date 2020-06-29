import time
import six
import os
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import serializers

class MLP(chainer.Chain):
    def __init__(self,n_uints,n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None,n_uints)
            self.l2 = L.Linear(None,n_uints)
            self.l3 = L.Linear(None,n_out)
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y
class SoftmaxClassifier(chainer.Chain):
    def __init__(self,predictor):
        super(SoftmaxClassifier, self).__init__(predictor=predictor)
    def __call__(self,x,t):
        y = self.predictor(x)
        self.loss = F.softmax_cross_entropy(y,t)
        self.accuracy = F.accuracy(y,t)
        return self.loss
if __name__ == '__main__':
    unit = 50
    class_num = 10
    gpu = 0
    batchsize = 100
    epoches = 10
    outdir = 'result'
    print('GPU:{}'.format(gpu))
    print('# uint:{}'.format(unit))
    print('# Minibatch-size:{}'.format(batchsize))
    print('# epoches:{}'.format(epoches))
    model = MLP(unit,class_num)
    classifier_model = SoftmaxClassifier(model)
    if gpu >=0:
        chainer.cuda.get_device(gpu).use()
        classifier_model.to_gpu()
    xp = np if gpu <0 else cuda.cupy
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(classifier_model)
    train,test = chainer.datasets.get_mnist()
    n_eopch = epoches
    N = len(train)
    N_test = len(test)
    for eopch in range(1,n_eopch+1):
        print('epoch:',eopch)
        perm = np.random.permutation(N)
        sum_accuracy = 0
        sum_loss = 0
        start = time.time()
        for i in six.moves.range(0,N,batchsize):
            x = chainer.Variable(xp.asarray(train[perm[i:i+batchsize]][0]))
            t = chainer.Variable(xp.asarray(train[perm[i:i+batchsize]][1]))
            optimizer.update(classifier_model,x,t)
            sum_loss += float(classifier_model.loss.data) * len(t.data)
            sum_accuracy += float(classifier_model.accuracy.data) * len(t.data)
        end = time.time()
        elapsed_time = end -start
        throughput = N/elapsed_time
        print('train mean loss={}, accuracy={}, throughput={} images/sec'.format(
            sum_loss / N, sum_accuracy / N, throughput))
        # evaluation
        sum_accuracy = 0
        sum_loss = 0
        for i in six.moves.range(0, N_test, batchsize):
            index = np.asarray(list(range(i, i + batchsize)))
            x = chainer.Variable(xp.asarray(test[index][0]))
            t = chainer.Variable(xp.asarray(test[index][1]))

            loss = classifier_model(x, t)
            sum_loss += float(loss.data) * len(t.data)
            sum_accuracy += float(classifier_model.accuracy.data) * len(t.data)

        print('test  mean loss={}, accuracy={}'.format(
            sum_loss / N_test, sum_accuracy / N_test))
    # Save the model and the optimizer
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    print('save the model')
    serializers.save_npz('{}/classifier_mlp.model'.format(outdir), classifier_model)
    serializers.save_npz('{}/mlp.model'.format(outdir), model)
    print('save the optimizer')
    serializers.save_npz('{}/mlp.state'.format(outdir), optimizer)






