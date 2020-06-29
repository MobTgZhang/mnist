import os
import tensorflow as tf
import datetime
def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    return model
@tf.function
def train_step(model,loss_func,optimizer,train_loss,train_metric,features,labels):
    with tf.GradientTape() as tape:
        predictions = model(features,training = True)
        loss = loss_func(labels,predictions)
    gradients = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))

    train_loss.update_state(loss)
    train_metric.update_state(labels,predictions)
@tf.function
def test_step(model,loss_func,test_loss,test_metric,features,labels):
    predictions = model(features)
    batch_loss = loss_func(labels,predictions)
    test_loss.update_state(batch_loss)
    test_metric.update_state(labels,predictions)
def train_model(model,loss_func,optimizer,epoches,
                train_loss,train_metric,test_loss,test_metric,
                ds_train,ds_test,
                summary_writer):
    for epoch in tf.range(1,epoches + 1,dtype='int64'):
        for features,lablels in ds_train:
            train_step(model,loss_func,optimizer,train_loss,train_metric,features,lablels)
        for features,lablels in ds_test:
            test_step(model,loss_func,test_loss,test_metric,features,lablels)
        with summary_writer.as_default():
            tf.summary.scalar("train loss", train_loss.result().numpy(), step=epoch)
            tf.summary.scalar("train accuracy", train_metric.result().numpy(), step=epoch)
            tf.summary.scalar("test loss", test_loss.result().numpy(), step=epoch)
            tf.summary.scalar("test accuracy", test_metric.result().numpy(), step=epoch)
        train_loss.reset_states()
        test_loss.reset_states()
        train_metric.reset_states()
        test_metric.reset_states()
if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        gpu0 = gpus[0]
        tf.config.experimental.set_memory_growth(gpu0,True)
        tf.config.set_visible_devices([gpu0],"GPU")
    tf.keras.backend.set_floatx('float64')
    (x_trian,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
    x_trian = tf.keras.utils.normalize(x_trian,axis=1)
    x_test = tf.keras.utils.normalize(x_test,axis=1)
    batchsize = 32
    epoches = 10
    ds_train = tf.data.Dataset.from_tensor_slices((x_trian,y_train))\
        .shuffle(buffer_size=1000).batch(batchsize)\
        .prefetch(tf.data.experimental.AUTOTUNE).cache()
    ds_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))\
        .shuffle(buffer_size = 1000).batch(batchsize)\
        .prefetch(tf.data.experimental.AUTOTUNE).cache()
    tf.keras.backend.clear_session()
    model = create_model()
    optimzier = tf.keras.optimizers.Nadam()
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_metric = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_metric = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = 'logs'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    log_dir = os.path.join(save_path,current_time)
    model_dir = os.path.join(save_path,current_time+"_mnist.h5")
    summary_writer = tf.summary.create_file_writer(log_dir)
    train_model(model,loss_func,optimzier,epoches,
                train_loss,train_metric,test_loss,test_metric,
                ds_train,ds_test,summary_writer)
    model.save(model_dir)

