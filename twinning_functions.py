from twinning_cpp import twin_cpp
import numpy as np

def twin():

	N = 1000
	p = 5
	rho = 0.5
	mu = np.zeros(p)
	sigma = np.zeros((p, p))

	for i in range(p):
		for j in range(p):
			sigma[i, j] = np.power(rho, np.abs(i - j))


	np.random.seed(25)
	data = np.random.multivariate_normal(mean=mu, cov=sigma, size=N)

	print(data[0, 3])
	print(data[2, 0])
	vec = twin_cpp(data)
	print(vec)
	print(type(vec))
	print(vec.dtype)




		