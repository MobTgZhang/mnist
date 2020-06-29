import numpy as np
import paddle
import paddle.dataset.mnist as mnist
import paddle.fluid as fluid
from PIL import Image
import os

def multilayer_perceptron(input):
    hidden1 = fluid.layers.fc(input=input,size=100,act='relu')
    hidden2 = fluid.layers.fc(input=hidden1,size=100,act='relu')
    fc = fluid.layers.fc(input=hidden2,size=10,act='softmax')
    return fc
def convolutional_neural_network(input):
    conv_1 = fluid.layers.conv2d(input=input,num_filters=32,filter_size=3,stride=1)
    pool_1 = fluid.layers.pool2d(input=conv_1,pool_size=2,pool_stride=1,pool_type='max')
    conv_2 = fluid.layers.conv2d(input=pool_1,num_filters=32,filter_size=3,stride=1)
    pool_2 = fluid.layers.pool2d(input=conv_2,pool_size=2,pool_stride=1,pool_type='max')
    fc = fluid.layers.fc(input=pool_2,size=10,act='softmax')
    return fc
def load_image(file):
    im = Image.open(file).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
    im = im / 255.0 * 2.0 - 1.0  # 归一化
    # mnist数据集中的图片默认格式是黑底白字，所以如果有需要则要进行变换
    return im
if __name__ == '__main__':
    image = fluid.layers.data(name='image',shape=[1,28,28],dtype='float32')
    label = fluid.layers.data(name='label',shape=[1],dtype='int64')
    result = convolutional_neural_network(image)
    cost = fluid.layers.cross_entropy(input=result,label=label)
    avg_cost = fluid.layers.mean(cost)
    accuracy = fluid.layers.accuracy(input=result,label=label)

    test_program = fluid.default_main_program().clone(for_test=True)

    optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.001)
    opts = optimizer.minimize(avg_cost)
    train_reader = paddle.batch(mnist.train(),batch_size=128)
    test_reader = paddle.batch(mnist.test(),batch_size=128)

    gpu = fluid.CUDAPlace(0)
    exe = fluid.Executor(gpu)
    exe.run(fluid.default_startup_program())
    feeder = fluid.DataFeeder(place=gpu,feed_list=[image,label])
    for pass_id in range(5):
        for batch_id,data in enumerate(train_reader()):
            train_cost,train_acc = exe.run(program=fluid.default_main_program(),
                                           feed=feeder.feed(data),
                                           fetch_list=[avg_cost,accuracy])
            if batch_id % 100 == 0:
                print('Pass:%d Batch:%d Cost:%0.5f Accuracy:%0.5f' % (pass_id, batch_id, train_cost[0], train_acc[0]))
        test_accs = []
        test_costs = []
        for batch_id, data in enumerate(test_reader()):
            test_cost, test_acc = exe.run(program=test_program, feed=feeder.feed(data), fetch_list=[avg_cost, accuracy])
            test_accs.append(test_acc[0])
            test_costs.append(test_cost[0])
        test_cost = (sum(test_costs) / len(test_costs))
        test_acc = (sum(test_accs) / len(test_accs))
        print('Test:%d Cost:%0.5f Accuracy:%0.5f' % (pass_id, test_cost, test_acc))

    # 保存预测模型，方便以后预测
    save_path = 'model/infer_model/'
    # 创建预测模型文件目录
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 保存预测模型,值得注意的是，这个需要把输入层的name记录下来以便以后使用输入层的name进行feed
    # 想要预测的内容也要存入模型，以便以后使用fetch_list进行预测
    fluid.io.save_inference_model(save_path, feeded_var_names=[image.name], target_vars=[result], executor=exe)
    # 保存参数模型
    save_path = 'model/params_model/'
    # 创建保持模型文件目录
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 保存模型参数
    fluid.io.save_params(executor=exe, dirname=save_path)
    # 保存持久化变量模型
    save_path = 'model/persistables_model/'
    # 创建保持模型文件目录
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 保存持久化变量模型
    fluid.io.save_persistables(executor=exe, dirname=save_path)


















