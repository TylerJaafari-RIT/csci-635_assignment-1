# Steps to follow:
#	I.	Preprocessing: Load dataset, Clean and remove rows with missing values, na if any, encoding if needed, 
# 		train/test split, scaling, etc.
#	II.	Implement and train GD variants using any appropriate hyperparameters. Make separate functions for each
# 		algorithm. State your assumptions in comments, if any. All libraries are allowed except 
# 		sklearn.linear_model.LinearRegression or any solver that directly returns β for you.

import numpy as np
import torch
import pandas as pd

from ucimlrepo import fetch_ucirepo
from ucimlrepo import dotdict

class BatchGradientDescent:
	def __init__(self, X):
		# HYPERPARAMETERS
		self.beta = np.zeros((X.shape[1], 1))
		self.eta = 0.05
		self.max_iter = 5000

	def gradient(self, beta, X, y):
		grad = -2 * (X.T @ y) + 2 * (X.T @ (X @ beta))
		return grad

	def build_model(self, X, y):
		for _ in range(self.max_iter):
			grad = self.gradient(self.beta, X, y)
			if np.any(np.isnan(grad)):
				print("NaN reached; stopping descent.")
				break
			self.beta -= self.eta * grad

		self.y_hat = X @ self.beta
		residuals = y - self.y_hat
		self.RSS = (residuals.T @ residuals)

		print(f"X: {X}")
		print(f"y: {y}")
		print(f"beta: {self.beta}")
		print(f"ŷ: {self.y_hat}")
		print(f"RSS: {self.RSS}")

class StochasticGradientDescent:
	def __init__(self, X, eta=0.01, max_epochs=100):
		# HYPERPARAMETERS
		self.beta = np.zeros((X.shape[1], 1))
		self.eta = eta
		self.max_epochs = max_epochs

	def gradient(self, beta, X, y):
		X = X.reshape(1, -1)
		return -2 * (X.T * (y - (X @ beta)).item())

	def build_model(self, X, y):
		n, p = X.shape
		for epoch in range(self.max_epochs):
			perm = np.random.permutation(n)
			for idx in perm:
				x_i = X[idx, :]
				if y.ndim == 2:
					y_i = y[idx, 0]
				else:
					y_i = y[idx]
				grad = self.gradient(x_i, y_i, self.beta)
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
			print(f"Epoch: {epoch}")
			print(f"RSS: {self.RSS}")

		self.y_hat = X @ self.beta
		residuals = y - self.y_hat
		self.RSS = (residuals.T @ residuals)

		print(f"beta: {self.beta}")
		print(f"ŷ: {self.y_hat}")
		print(f"RSS: {self.RSS}")


class MiniBatchGradientDescent:
	def __init__(self, X):
		# HYPERPARAMETERS
		self.beta = np.zeros((X.shape[1], 1))
		self.eta = 0.05
		self.batch_size = 0.2 # defined as a portion of the dataset size
		self.max_epochs = 5
		self.max_iter = 500

	def gradient(self, beta, X, y):
		grad = -2 * (X.T @ y) + 2 * (X.T @ (X @ beta))
		return grad

	def build_model(self, X, y):
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
		print(f"beta: {self.beta}")
		print(f"ŷ: {self.y_hat}")
		print(f"RSS: {self.RSS}")

def main():
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

	# data_output = open('auto_mpg.txt', mode='w')
	# print(X.to_string(), file=data_output)
	# print('\n\n', file=data_output)
	# print(y.to_string(), file=data_output)
	# data_output.close()

	combined_output = open('combined.txt', mode='w')
	print(combined.to_string(), file=combined_output)
	combined_output.close()

	X = combined.drop(columns='mpg')
	y = combined.drop(columns=X.columns)

	# metadata
	print(auto_mpg.metadata)

	# variable information
	print(auto_mpg.variables)
			
	print(f"X: {X}")
	# bgd = BatchGradientDescent(np.asarray(X))
	# bgd.build_model(np.asarray(X), y)

	# sgd = StochasticGradientDescent(np.asarray(X))
	# sgd.build_model(np.asarray(X), np.asarray(y))

	mbgd = MiniBatchGradientDescent(np.asarray(X))
	mbgd.build_model(np.asarray(X), np.asarray(y))

if __name__ == "__main__":
	main()