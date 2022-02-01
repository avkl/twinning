# Data Twinning
A ``Python3`` module for data twinning. For ``R``, the package is available from <a href="https://cran.r-project.org/web/packages/twinning/index.html" target="_blank">CRAN</a>.

## About
An efficient implementation of the twinning algorithm proposed in Vakayil and Joseph (2022) for partitioning a dataset into statistically similar twin sets. The algorithm is orders of magnitude faster than the ``SPlit`` algorithm proposed in Joseph and Vakayil (2021) for optimally splitting a dataset into training and testing sets, and the ``support points`` algorithm of Mak and Joseph (2018) for subsampling from Big Data.

The module provides functions ``twin()``, ``multiplet()``, and ``energy()``. 

- ``twin()`` partitions datasets into statistically similar disjoint sets, termed as *twins*. The twins themselves are statistically similar to the original dataset (Vakayil and Joseph, 2022). Such a partition can be employed for optimal training and testing of statistical and machine learning models (Joseph and Vakayil, 2021). The twins can be of unequal size; for tractable model building on large datasets, the smaller twin can serve as a compression (lossy) of the original dataset. 

- ``multiplet()`` is an extension of ``twin()`` to generate multiple disjoint partitions that can be used for *k*-fold cross validation, or with divide-and-conquer procedures.

- ``energy()`` computes the energy distance (Székely and Rizzo, 2013) between a given dataset and a set of points, which is the metric minimized by twinning.

This work is supported by U.S. National Science Foundation grants **DMREF-1921873** and **CMMI-1921646**.

## Installation
  ```shell
  > pip3 install git+https://github.com/avkl/twinning.git
  ```

## How to use
### ``twin()``
``twin()`` accepts a numpy ndarray as the dataset, and an integer parameter ``r`` representing the inverse of the partitioning ratio, i.e., for an 80-20 split, ``r`` = 1 / 0.2 = 5. The function returns the indices of the smaller twin.

The following code generates an 80-20 partition of a dataset with two columns. The encircled points in the figure depict the smaller twin.


```python
import numpy as np
import matplotlib.pyplot as plt

x = np.random.normal(loc=0, scale=1, size=100)
y = np.random.normal(loc=np.power(x, 2), scale=1, size=100)
data = np.hstack((x.reshape(100, 1), y.reshape(100, 1)))
twin_idx = twin(data, r=5)

plt.scatter(x, y, alpha=0.5)
plt.scatter(x[twin_idx], y[twin_idx], s=125, facecolors="none", edgecolors="black")
```
![twinning](https://raw.githubusercontent.com/avkl/main/html/twinning.png)