from twinning_cpp import twin_cpp, multiplet_S3_cpp, energy_cpp
import numpy as np
import math


def data_format(data):
	const_cols = np.all(data == data[0, :], axis=0)
	data = data[:, np.invert(const_cols)]
	data = (data - data.mean(axis=0)) / data.std(axis=0)
	
	if data.data.c_contiguous:
		return data
	else:
		return np.copy(data, order='C')


def twin(data, r, u1=None, leaf_size=8):
	if type(data) != np.ndarray or len(data.shape) != 2:
		raise Exception("data is expected to be a 2 dimensional numpy ndarray")

	if np.isnan(data).any() or np.isinf(data).any():
		raise Exception("data cannot contain nan or infinity")

	if u1 is None:
		u1 = np.random.randint(data.shape[0])
	elif u1 not in range(data.shape[0]):
		raise Exception("u1 should be a row index such that 0 <= u1 < data.shape[0]")

	if r not in range(2, math.floor(data.shape[0] / 2) + 1):
		raise Exception("r should be an integer such that 2 <= r <= data.shape[0]/2")
	
	data = data_format(data)
	return np.array(twin_cpp(data, r, u1, leaf_size), dtype='uint64')


def multiplet(data, k, strategy=1, leaf_size=8):
	if type(data) != np.ndarray or len(data.shape) != 2:
		raise Exception("data is expected to be a 2 dimensional numpy ndarray")

	if np.isnan(data).any() or np.isinf(data).any():
		raise Exception("data cannot contain nan or infinity")

	if k not in range(2, math.floor(data.shape[0] / 2) + 1):
		raise Exception("k should be an integer such that 2 <= r <= data.shape[0]/2")

	data = data_format(data)
	N = data.shape[0]

	if strategy == 1:
		row_index = np.arange(N)
		folds = np.empty((0, 2))
		i = 0
		while True:
			multiplet_i = np.array(twin_cpp(data, k - i, np.random.randint(data.shape[0]), leaf_size), dtype='uint64')
			fold = np.hstack((row_index[multiplet_i].reshape(len(multiplet_i), 1), np.repeat(i, len(multiplet_i)).reshape(len(multiplet_i), 1)))
			folds = np.vstack((folds, fold))
			
			negate = np.ones(data.shape[0], bool)
			negate[multiplet_i] = 0
			data = data[negate, :]
			row_index = row_index[negate]

			if data.shape[0] <= N / k:
				fold = np.hstack((row_index.reshape(len(row_index), 1), np.repeat(i + 1, len(row_index)).reshape(len(row_index), 1)))
				folds = np.vstack((folds, fold))
				break

			i += 1

		return folds[np.argsort(folds[:, 0]), 1].astype('uint64')

	if strategy == 2:
		if not (k & (k - 1) == 0):
			raise Exception("strategy 2 requires k to be a power of 2")

		row_index = np.arange(N)
		folds = np.empty((0, 2))
		i = 0

		def equal_twins(data, row_index):
			if data.shape[0] <= math.ceil(N / k):
				nonlocal folds, i
				fold = np.hstack((row_index.reshape(len(row_index), 1), np.repeat(i, len(row_index)).reshape(len(row_index), 1)))
				folds = np.vstack((folds, fold))
				i += 1
			else:
				equal_twins_i = np.array(twin_cpp(data, 2, np.random.randint(data.shape[0]), leaf_size), dtype='uint64')
				negate = np.ones(data.shape[0], bool)
				negate[equal_twins_i] = 0
				equal_twins(data[negate, :], row_index[negate])
				equal_twins(data[np.invert(negate), :], row_index[np.invert(negate)])

		equal_twins(data, row_index)
		return folds[np.argsort(folds[:, 0]), 1].astype('uint64')

	if strategy == 3:
		sequence = np.array(multiplet_S3_cpp(data, k, np.random.randint(data.shape[0]), leaf_size), dtype='uint64')
		folds = np.hstack((sequence.reshape(len(sequence), 1), np.tile(np.arange(k), np.ceil(N / k).astype('uint64'))[0:N].reshape(N, 1)))
		return folds[np.argsort(folds[:, 0]), 1].astype('uint64')


def energy(data, points):
	if type(data) != np.ndarray or len(data.shape) != 2:
		raise Exception("data is expected to be a 2 dimensional numpy ndarray")

	if np.isnan(data).any() or np.isinf(data).any():
		raise Exception("data cannot contain nan or infinity")

	if type(points) != np.ndarray or len(points.shape) != 2:
		raise Exception("points is expected to be a 2 dimensional numpy ndarray")

	if np.isnan(points).any() or np.isinf(points).any():
		raise Exception("points cannot contain nan or infinity")

	if data.shape[1] != points.shape[1]:
		raise Exception("data and points should have the same number of columns")

	const_cols = np.all(data == data[0, :], axis=0)
	data = data[:, np.invert(const_cols)]
	points = points[:, np.invert(const_cols)]

	data_mean = data.mean(axis=0)
	data_std = data.std(axis=0)
	data = (data - data_mean) / data_std
	points = (points - data_mean) / data_std

	if not data.data.c_contiguous:
		data = np.copy(data, order='C')

	if not points.data.c_contiguous:
		points = np.copy(points, order='C')

	return energy_cpp(data, points)



