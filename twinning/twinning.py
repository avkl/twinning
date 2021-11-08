from twinning_cpp import twin_cpp
import numpy as np

def twin(data):
	return(np.array(twin_cpp(data, 5, np.random.randint(data.shape[0]), 8), dtype='uint64'))


def multiplet(data, n):
	N = data.shape[0];
	d = data.shape[1];
	data_ = np.hstack((np.array(range(N)).reshape(N, 1), data))

	folds = np.empty((0, 2))
	D_ = data_
	i = 0
	while True:
		negate = np.ones(D_.shape[0], np.bool)
		multiplet_i = np.array(twin_cpp(np.copy(D_[:, 1:], order='C'), n - i, np.random.randint(D_.shape[0]), 8), dtype='uint64')
		
		fold = np.hstack((D_[multiplet_i, 0].reshape(len(multiplet_i), 1), np.repeat(i, len(multiplet_i)).reshape(len(multiplet_i), 1)))
		folds = np.vstack((folds, fold))
		negate[multiplet_i] = 0
		D_ = D_[negate, :]

		if D_.shape[0] <= N / n:
			fold = np.hstack((D_[:, 0].reshape(D_.shape[0], 1), np.repeat(i + 1, D_.shape[0]).reshape(D_.shape[0], 1)))
			folds = np.vstack((folds, fold))
			break

		i += 1

	return(folds[np.argsort(folds[:, 0]), 1].astype('uint64'))













		