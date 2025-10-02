# Steps to follow:
#	I.	Preprocessing: Load dataset, Clean and remove rows with missing values, na if any, encoding if needed, 
# 		train/test split, scaling, etc.
#	II.	Implement and train GD variants using any appropriate hyperparameters. Make separate functions for each
# 		algorithm. State your assumptions in comments, if any. All libraries are allowed except 
# 		sklearn.linear_model.LinearRegression or any solver that directly returns β for you.

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ucimlrepo import fetch_ucirepo

class BatchGradientDescent:
	def __init__(self, X, eta=0.005, max_iter=100):
		# HYPERPARAMETERS
		self.beta = np.zeros((X.shape[1], 1))
		self.eta = eta
		self.max_iter = max_iter

	def gradient(self, beta, X, y):
		grad = -2 * (X.T @ y) + 2 * (X.T @ (X @ beta))
		return grad

	def build_model(self, X, y):
		result_output = open('BGDresults.rtf', mode='w')
		for _ in range(self.max_iter):
			grad = self.gradient(self.beta, X, y)
			if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
				print("Convergence reached; stopping descent.")
				break
			self.beta -= self.eta * grad

		self.y_hat = X @ self.beta
		residuals = y - self.y_hat
		self.RSS = (residuals.T @ residuals)

		print(f"X: {X}", file=result_output)
		print(f"y: {y}", file=result_output)
		print(f"beta: {self.beta}", file=result_output)
		print(f"y_hat: {self.y_hat}", file=result_output)
		print(f"RSS: {self.RSS}", file=result_output)

class StochasticGradientDescent:
	def __init__(self, X, eta=0.005, max_epochs=50):
		# HYPERPARAMETERS
		self.beta = np.zeros((X.shape[1], 1))
		self.eta = eta
		self.max_epochs = max_epochs

	def gradient(self, beta, X, y):
		sample = X.reshape(1, -1)
		return -2 * (sample.T * (y - (sample @ beta)).item())

	def build_model(self, X, y):
		result_output = open('SGDresults.rtf', mode='w')
		n, p = X.shape
		for epoch in range(self.max_epochs):
			perm = np.random.permutation(n)
			for idx in perm:
				x_i = X[idx, :]
				if y.ndim == 2:
					y_i = y[idx, 0]
				else:
					y_i = y[idx]
				grad = self.gradient(self.beta, x_i, y_i)
				if np.any(np.isnan(grad)):
					print(f"NaN reached at epoch {epoch}, idx {idx}")
					return
				self.beta -= self.eta * grad
				if np.any(np.isnan(self.beta)):
					print(f"NaN reached at epoch {epoch}, idx {idx}")
					return

			y_hat = X @ self.beta
			residuals = y - y_hat
			self.RSS = (residuals.T @ residuals).item()
			print(f"Epoch: {epoch}", file=result_output)
			print(f"RSS: {self.RSS}", file=result_output)

		self.y_hat = X @ self.beta
		residuals = y - self.y_hat
		self.RSS = (residuals.T @ residuals)

		print(f"beta: {self.beta}", file=result_output)
		print(f"y_hat: {self.y_hat}", file=result_output)
		print(f"RSS: {self.RSS}", file=result_output)


class MiniBatchGradientDescent:
	def __init__(self, X):
		# HYPERPARAMETERS
		self.beta = np.zeros((X.shape[1], 1))
		self.eta = 0.005
		self.batch_size = 0.2 # defined as a portion of the dataset size
		self.max_epochs = 5
		self.max_iter = 50

	def gradient(self, beta, X, y):
		grad = -2 * (X.T @ y) + 2 * (X.T @ (X @ beta))
		return grad

	def build_model(self, X, y):
		result_output = open('MBGDresults.rtf', mode='w')
		for epoch in range(self.max_epochs):
			# TODO: work in mean gradient over iterations
			batch_size = int(self.batch_size * X.shape[0])
			start_point = np.random.randint(0, X.shape[0] - batch_size)
			mini_batch = X[start_point : start_point + batch_size]
			batch_targets = y[start_point : start_point + batch_size]
			print(f"Batch Size: {batch_size} rows")
			print(f"Mini Batch: {mini_batch}")
			print(f"Batch Target: {batch_targets}")
			for _ in range(self.max_iter):
				grad = self.gradient(self.beta, mini_batch, batch_targets)
				if np.any(np.isnan(grad)):
					print(f"NaN reached at epoch {epoch}; stopping descent.")
					break
				self.beta -= self.eta * grad

		self.y_hat = X @ self.beta
		residuals = y - self.y_hat
		self.RSS = (residuals.T @ residuals)

		# print(f"X: {X}")
		# print(f"y: {y}")
		print(f"beta: {self.beta}", file=result_output)
		print(f"y_hat: {self.y_hat}", file=result_output)
		print(f"RSS: {self.RSS}", file=result_output)

def main():
	np.set_printoptions(threshold=100000)
	# fetch dataset
	auto_mpg = fetch_ucirepo(id=9)

	# data (as pandas dataframes)
	X : pd.DataFrame = auto_mpg.data.features
	y : pd.DataFrame = auto_mpg.data.targets

	# rows_diff = X.shape[0]
	# X = X.dropna()
	# rows_diff = rows_diff - X.shape[0]

	# y = y[0:-rows_diff]
	combined = pd.concat((X, y), axis=1)
	combined = combined.dropna()

	combined_output = open('combined.txt', mode='w')
	print(combined.to_string(), file=combined_output)
	combined_output.close()

	X = combined.drop(columns='mpg')
	y = combined.drop(columns=X.columns)

	X = StandardScaler().fit_transform(X, y)
	y = StandardScaler().fit_transform(y)

	data_output = open('split.txt', mode='w')
	print(X, file=data_output)
	print('\n\n', file=data_output)
	print(y, file=data_output)
	data_output.close()

	# metadata
	print(auto_mpg.metadata)

	# variable information
	print(auto_mpg.variables)

	print(f"X: {X}")
	bgd = BatchGradientDescent(X)
	bgd.build_model(X, y)

	sgd = StochasticGradientDescent(X)
	sgd.build_model(X, y)

	mbgd = MiniBatchGradientDescent(X)
	mbgd.build_model(X, y)

if __name__ == "__main__":
	main()