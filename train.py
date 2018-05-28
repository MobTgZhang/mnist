from model import BpNet,cross_entropy_cost
from utils import DataSet,test_data_op,process_label,load_mnist
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import numpy as np
import pickle 
import os
def train_bpnet():
	path = "mnist"
	# dataset preparation
	X_train,y_train = load_mnist(path,"train")
	X_test,y_test = load_mnist(path,"t10k")
	# dataset defination
	train_data = DataSet(X_train,y_train)
	test_data = DataSet(X_test,y_test)
	# parameters defination
	in_size = 784
	hid_size = 100
	out_size = 10
	batch_size = 20
	epoches = 150
	# model defination
	net = BpNet(in_size,hid_size,out_size)
	print(train_data.input_data.shape,train_data.target_data.shape)
	# determine inputs dtype 
	x = T.dmatrix("x")
	y = T.dmatrix("y")

	# defination of the layers 
	learning_rate = 0.1

	prediction = net.forward(x)
	# cost function defination
	cost = cross_entropy_cost(prediction,y)
	# update the grad 
	list_q = net.update_grad(cost,learning_rate)
	
	# apply gradient descent
	train = theano.function(
		inputs = [x,y],
		outputs = [cost],
		updates = list_q
		)
	# prediction 
	predict = theano.function(inputs = [x],outputs = prediction)
	# training model 
	loss_list = []
	percentage_list = []
	for k in range(epoches):
		Length = len(X_train) // batch_size
		sum_loss = 0
		for j in range(Length):
			out_x = X_train[j*batch_size:(j+1)*batch_size,:]
			out_y = y_train[j*batch_size:(j+1)*batch_size,:]
			err = train(out_x,out_y)
			sum_loss += err[0]
		out_pre = predict(test_data.input_data)
		out_org = test_data.target_data
		percentage = test_data_op(out_pre,out_org)
		print("epoches:%d loss:%0.4f correct:%0.2f%%"%(k,sum_loss/Length,percentage*100))
		loss_list.append(sum_loss/Length)
		percentage_list.append(percentage)
	# ----------------------------------------------------------------
	# save the model
	model_name = 'bpnet.pkt'
	path_save = 'bpnet'
	if not os.path.exists(path_save):
		os.mkdir(path_save)
	f= open(os.path.join(path_save,model_name), 'wb')
	pickle.dump(net,f, protocol=pickle.HIGHEST_PROTOCOL)  
	f.close()
	# ----------------------------------------------------------------
	# save the loss image
	x = np.linspace(0,len(loss_list),len(loss_list))
	plt.plot(x,loss_list)
	plt.savefig(os.path.join(path_save,"loss.png"))
	plt.show()
	with open(os.path.join(path_save,"loss.txt"),"w") as fp:
		for k in range(len(loss_list)):
			fp.write(str(loss_list[k]) + "\n")
	x = np.linspace(0,len(percentage_list),len(percentage_list))
	plt.plot(x,percentage_list)
	plt.savefig(os.path.join(path_save,"percentage.png"))
	plt.show()
	with open(os.path.join(path_save,"percentage.txt"),"w") as fp:
		for k in range(len(percentage_list)):
			fp.write(str(percentage_list[k]) + "\n")
if __name__ == "__main__":
	train_bpnet()