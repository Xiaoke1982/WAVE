import numpy as np
from sklearn.tree import DecisionTreeClassifier

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
		else:
			if self.base_ensemble == "rf":
			    max_features = "sqrt"
		    elif self.base_ensemble == "bagging":
			    max_features = None
		    else:
			    raise ValueError("not a valid ensemble type")
		
			for i in range(self.ensemble_size):
				idxes = np.random.choice(np.arange(len(train_X)), size=len(train_X), replace=True)
				bootstrap_X = train_X[idxes]
				bootstrap_y = train_y[idxes]
				
				tree = DecisionTreeClassifier(random_state=self.random_state, max_features=max_features)
				tree.fit(bootstrap_X, bootstrap_y)
				self.base_classifiers.append(tree)
	
	def compute_weights(self, train_X, train_y):
		
	
	
	def predict(self, new_X, return_type="label")