import numpy as np
import struct
import os

def process_label(input_V):
	out_s = np.zeros((len(input_V),10),dtype = np.float32)
	for k in range(len(input_V)):
		out_s[k][input_V[k]] = 1
	return out_s
def load_mnist(path, kind='t10k'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)
    labels = process_label(labels)
    return images, labels
class DataSet:
	def __init__(self,input_data,target_data):
		# processing target data
		self.target_data = target_data
		self.input_data = input_data
	def __getitem__(self,index):
		return (self.input_data[index],self.target_data[index])
	def __len__(self):
		return len(self.target_data)
def test_data_op(out_pre,out_org):
  labels_test_pre = []
  labels_test_org = []
  Length = len(out_org)
  for k in range(Length):
    e0 = out_pre[k]
    s0 = np.where(e0 == np.max(e0))
    labels_test_pre.append(s0[0][0])
    e1 = out_org[k]
    s1 = np.where(e1 == np.max(e1))
    labels_test_org.append(s1[0][0])
  out = [1 if labels_test_pre[k] == labels_test_org[k] else 0 for k in range(Length)]
  return sum(out)/len(out)