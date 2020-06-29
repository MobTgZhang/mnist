import mxnet as mx
def build_graph(data):
    fc1 = mx.symbol.FullyConnected(data,name='fc1',num_hidden=128)
    act1 = mx.symbol.Activation(fc1,name='relu1',act_type='relu')
    fc2 = mx.symbol.FullyConnected(act1,name='fc2',num_hidden=64)
    act2 = mx.symbol.Activation(fc2,name='relu2',act_type='relu')
    fc3 = mx.symbol.FullyConnected(act2,name='fc3',num_hidden=10)
    softmax = mx.symbol.SoftmaxOutput(fc3,name='softmax')
    return softmax
if __name__ == '__main__':
    batchsize = 100
    n_epoch = 5
    # Use GPU if available
    if len(mx.test_utils.list_gpus()) != 0:  # return range(0, 4)
        ctx = mx.gpu()  # default gpu(0)
    else:
        ctx = mx.cpu()

    mnist = mx.test_utils.get_mnist()
    train_iter = mx.io.NDArrayIter(mnist['train_data'],mnist['train_label'],batchsize,shuffle=True)
    test_iter = mx.io.NDArrayIter(mnist['test_data'],mnist['test_label'],batchsize)

    data = mx.symbol.Variable(name='data')
    softmax = build_graph(data)
    mod = mx.mod.Module(softmax)
    mod.bind(data_shapes=train_iter.provide_data,label_shapes=train_iter.provide_label)
    mod.init_params()
    mod.init_optimizer(optimizer_params={'learning_rate':0.01,'momentum':0.9})
    train_m = mx.metric.create('acc')
    test_m = mx.metric.create('acc')
    for epoch in range(n_epoch):
        for i_iter,batch in enumerate(train_iter):
            mod.forward(batch)
            mod.update_metric(train_m,batch.label)
            mod.backward()
            mod.update()
        for name,val in train_m.get_name_value():
            print('epoch %03d:%s=%f'%(epoch,name,val))
        train_m.reset()
        train_iter.reset()
        for i_iter,batch in enumerate(test_iter):
            mod.forward(batch)
            mod.update_metric(test_m,batch.label)
        for name,val in test_m.get_name_value():
            print('epoch %03d:%s=%f'%(epoch,name,val))
        test_m.reset()
        test_iter.reset()

















