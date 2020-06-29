import theano 
import theano.tensor as T
import numpy as np
class linear:
	def __init__(self,in_size,out_size,activity_function = None):
		self.input_size = in_size
		self.output_size = out_size
		self.weight = theano.shared(np.random.normal(0,1,(in_size,out_size)))
		self.bais = theano.shared(np.zeros((out_size,)) + 0.1)
		self.act_func = activity_function
		# gradient 
		self.gW = None
		self.gb = None
	def forward(self,x):
		W_plus_b = T.dot(x,self.weight) + self.bais
		if self.act_func is None:
			return W_plus_b
		else:
			return self.act_func(W_plus_b)
	def update_grad(self,cost):
		self.gW,self.gb = T.grad(cost,[self.weight,self.bais])
		return (self.weight,self.gW),(self.bais,self.gb)
class BpNet:
	def __init__(self,input_size,hidden_size,output_size,activity_function = None):
		if activity_function is not None:
			self.act_func = activity_function
		else:
			self.act_func = T.tanh
		self.hidden = linear(input_size,hidden_size,self.act_func)
		self.predict = linear(hidden_size,output_size,None)
	def forward(self,x):
		temp = self.hidden.forward(x)
		temp = self.predict.forward(temp)
		return T.nnet.softmax(temp)
	def update_grad(self,cost,learning_rate):
		temp = []
		tupleH = []
		listA,listB = self.hidden.update_grad(cost)
		tupleH.append(listA)
		tupleH.append(listB)
		listA,listB = self.predict.update_grad(cost)
		tupleH.append(listA)
		tupleH.append(listB)
		for k in range(len(tupleH)):
			temp.append((tupleH[k][0],tupleH[k][0]-learning_rate * tupleH[k][1]))
		return temp
def cross_entropy_cost(y_target,y_label):
	cost = - (y_label*T.log(y_target)+(1-y_label)*T.log(1-y_target)).mean()
	return cost