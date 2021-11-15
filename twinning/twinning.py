from twinning_cpp import twin_cpp
import numpy as np

def data_format(data):
	col_cleanup = []
	for j in range(data.shape[1]):
	    if np.max(data[:, j]) == np.min(data[:, j]):
	        col_cleanup.append(j)

	negate = np.ones(data.shape[1], bool)
	negate[col_cleanup] = 0
	data = data[:, negate]
	data = (data - data.mean(axis=0)) / data.std(axis=0)
	return np.copy(data, order='C')


def twin(data, r):
	D = data_format(data)
	return(np.array(twin_cpp(D, r, np.random.randint(data.shape[0]), 8), dtype='uint64'))


def multiplet(data, n):
	D = data_format(data)
	N = D.shape[0];
	row_index = np.array(range(N))
	folds = np.empty((0, 2))
	i = 0
	start = time.time()
	while True:
		multiplet_i = np.array(twin_cpp(D, n - i, np.random.randint(D.shape[0]), 8), dtype='uint64')
		fold = np.hstack((row_index[multiplet_i].reshape(len(multiplet_i), 1), np.repeat(i, len(multiplet_i)).reshape(len(multiplet_i), 1)))
		folds = np.vstack((folds, fold))
		
		negate = np.ones(D.shape[0], np.bool)
		negate[multiplet_i] = 0
		D = D[negate, :]
		row_index = row_index[negate]

		if D.shape[0] <= N / n:
			fold = np.hstack((row_index.reshape(len(row_index), 1), np.repeat(i + 1, len(row_index)).reshape(len(row_index), 1)))
			folds = np.vstack((folds, fold))
			break

		i += 1

	return(folds[np.argsort(folds[:, 0]), 1].astype('uint64'))













		