"""
Data Twinning
=============
An efficient implementation of the twinning algorithm proposed in Vakayil and Joseph (2022) for partitioning a dataset into statistically similar twin sets. It is orders of magnitude faster than the SPlit algorithm proposed in Joseph and Vakayil (2021) for splitting a dataset into training and testing sets, and the support points algorithm of Mak and Joseph (2018) for subsampling from Big Data.
The module provides functions twin(), multiplet(), and energy(). twin() partitions datasets into statistically similar disjoint sets, termed as twins. The twins themselves are statistically similar to the original dataset (Vakayil and Joseph, 2022). Such a partition can be employed for optimal training and testing of statistical and machine learning models (Joseph and Vakayil, 2021). The twins can be of unequal size; for tractable model building on large datasets, the smaller twin can serve as a compressed (lossy) version of the original dataset. multiplet() is an extension of twin() to generate multiple disjoint partitions that can be used for k-fold cross validation, or with divide-and-conquer procedures. energy() computes the energy distance (Székely and Rizzo, 2013) between a given dataset and a set of points, which is the metric minimized by twinning.

How to use
==========
Visit https://github.com/avkl/twinning for details on how to use the module

References
==========
Vakayil, A., & Joseph, V. R. (2022). Data Twinning. Statistical Analysis and Data Mining: The ASA Data Science Journal, to appear. arXiv preprint arXiv:2110.02927.
Joseph, V. R., & Vakayil, A. (2021). SPlit: An Optimal Method for Data Splitting. Technometrics, 1-11. doi:10.1080/00401706.2021.1921037.
Mak, S. & Joseph, V. R. (2018). Support Points. Annals of Statistics, 46, 2562-2592.
Székely, G. J., & Rizzo, M. L. (2013). Energy statistics: A class of statistics based on distances. Journal of statistical planning and inference, 143(8), 1249-1272.
"""

from .twinning import twin, multiplet, energy