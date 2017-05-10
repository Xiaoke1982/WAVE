import numpy as np
from sklearn import DecisionTree

class WAVE:
	
	def __init__(self, ensemble_size, base_ensemble="rf"):
		self.ensemble_size = ensemble_size
		self.base_ensemble = base_ensemble
		self.weights = None
		self.base_classifiers = []
		
	def fit(self, train_X, train_y):
		self.fit_base_classifiers(train_X, train_y)
		self.compute_weights(train_X, train_y)
		
		
	def fit_base_classifiers(self, train_X, train_y):
		if self.base_ensemble == "cerp":
			pass
		elif self.base_ensemble == "rf":
			pass
		elif self.base_ensemble == "bagging":
			pass
		else:
			pass
				
	
	def compute_weights(self, train_X, train_y):
		
	
	
	def predict(self, new_X, return_type="label")