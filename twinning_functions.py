from twinning_cpp import twin_cpp
import numpy as np

def twin(data):
	return(np.array(twin_cpp(data, 5, np.random.randint(data.shape[0]), 8), dtype='uint64'))




		