from twinning_cpp import twin_cpp, multiplet_S3_cpp, energy_cpp
import numpy as np
import math


def _data_format(data):
	const_cols = np.all(data == data[0, :], axis=0)
	data = data[:, np.invert(const_cols)]
	data = (data - data.mean(axis=0)) / data.std(axis=0)
	
	if data.data.c_contiguous:
		return data
	else:
		return np.copy(data, order='C')


def twin(data, r, u1=None, leaf_size=8):
	"""
	**Descritpion**

	``twin()`` implements the data twinning algorithm presented in Vakayil and Joseph (2022). A partition of the dataset is returned, such that the resulting two disjoint sets, termed as *twins*, are distributed similar to each other, as well as the whole dataset. Such a partition is an optimal training-testing split (Joseph and Vakayil, 2021) for training and testing statistical and machine learning models, and is model-independent. The statistical similarity also allows one to treat either of the twins as a compression (lossy) of the dataset for tractable model building on Big Data.

	**Parameters**

	``data`` ( ndarray ): the dataset including both the predictors and response(s); should not contain nan or infinity

	``r`` ( int ): an integer representing the inverse of the splitting ratio, e.g., for an 80-20 partition, ``r`` = 1 / 0.2 = 5

	``u1`` ( int , optional ): index of the data point from where twinning starts; if not provided, a random point is chosen from the dataset; fixing ``u1`` makes the algorithm deterministic, i.e., the same twins are returned

	``leaf_size`` ( int , optional ): maximum number of elements in the leaf-nodes of the kd-tree

	**Returns**

	( ndarray ): indices of the smaller twin

	**Details**

	Before twinning, constant columns are removed from ``data`` and the remaining are scaled to zero mean and unit standard deviation. Twinning algorithm requires nearest neighbor queries that are performed using a *kd*-tree. The *kd*-tree implementation in the nanoflann (Blanco and Rai, 2014) C++ library is used.

	**References**

	Vakayil, A., & Joseph, V. R. (2022). Data Twinning. Statistical Analysis and Data Mining: The ASA Data Science Journal. https://doi.org/10.1002/sam.11574

	Joseph, V. R., & Vakayil, A. (2021). SPlit: An Optimal Method for Data Splitting. Technometrics, 1-11. doi:10.1080/00401706.2021.1921037.

	Blanco, J. L. & Rai, P. K. (2014). nanoflann: a C++ header-only fork of FLANN, a library for nearest neighbor (NN) with kd-trees. https://github.com/jlblancoc/nanoflann.

	"""

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
	
	data = _data_format(data)
	return np.array(twin_cpp(data, r, u1, leaf_size), dtype='uint64')


def multiplet(data, k, strategy=1, leaf_size=8):
	"""
	**Descritpion**

	``multiplet()`` extends ``twin()`` to partition datasets into multiple statistically similar disjoint sets, termed as *multiplets*, under the three different strategies described in Vakayil and Joseph (2022).

	**Parameters**

	``data`` ( ndarray ): the dataset including both the predictors and response(s); should not contain nan or infinity

	``k`` ( int ): the desired number of multiplets

	``strategy`` ( int , optional ): an integer either 1, 2, or 3 referring to the three strategies for generating multiplets; strategy 2 perfroms best, but requires ``k`` to be a power of 2; strategy 3 is computatioanlly inexpensive, but performs worse than strategies 1 and 2

	``leaf_size`` ( int , optional ): maximum number of elements in the leaf-nodes of the kd-tree

	**Returns**

	( ndarray ): array with the multiplet id, ranging from 0 to ``k`` - 1, for each row in data

	**References**

	Vakayil, A., & Joseph, V. R. (2022). Data Twinning. Statistical Analysis and Data Mining: The ASA Data Science Journal. https://doi.org/10.1002/sam.11574

	Blanco, J. L. & Rai, P. K. (2014). nanoflann: a C++ header-only fork of FLANN, a library for nearest neighbor (NN) with kd-trees. https://github.com/jlblancoc/nanoflann.

	"""

	if type(data) != np.ndarray or len(data.shape) != 2:
		raise Exception("data is expected to be a 2 dimensional numpy ndarray")

	if np.isnan(data).any() or np.isinf(data).any():
		raise Exception("data cannot contain nan or infinity")

	if k not in range(2, math.floor(data.shape[0] / 2) + 1):
		raise Exception("k should be an integer such that 2 <= r <= data.shape[0]/2")

	data = _data_format(data)
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
	"""
	**Descritpion**

	``energy()`` computes the energy distance (Székely and Rizzo, 2013) between a given dataset and a set of points in same dimensions.

	**Parameters**

	``data`` ( ndarray ): the dataset including both the predictors and response(s); should not contain nan or infinity

	``points`` ( ndarray ): the set of points for which the energy distance with respect to ``data`` is to be computed; should not contain nan or infinity

	**Returns**

	( float ): energy distance

	**Details**

	Smaller the energy distance, the more statistically similar the set of points is to the given dataset. The minimizer of energy distance is known as support points (Mak and Joseph, 2018), which is the basis for the twinning method. Computing energy distance between ``data`` and ``points`` involves Euclidean distance calculations among the rows of ``data``, among the rows of ``points``, and between the rows of ``data`` and ``points``. Since, ``data`` serves as the reference, the distance calculations among the rows of ``data`` are ignored for efficiency. Before computing the energy distance, the columns of ``data`` are scaled to zero mean and unit standard deviation. The mean and standard deviation of the columns of ``data`` are used to scale the respective columns in ``points``.

	**References**

	Vakayil, A., & Joseph, V. R. (2022). Data Twinning. Statistical Analysis and Data Mining: The ASA Data Science Journal. https://doi.org/10.1002/sam.11574

	Székely, G. J., & Rizzo, M. L. (2013). Energy statistics: A class of statistics based on distances. Journal of statistical planning and inference, 143(8), 1249-1272.

	Mak, S. & Joseph, V. R. (2018). Support Points. Annals of Statistics, 46, 2562-2592.

	"""

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



