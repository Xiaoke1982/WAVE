import numpy as np
from sklearn.tree import DecisionTreeClassifier

class WAVE(object):
	
	def __init__(self, ensemble_size, base_ensemble="rf"):
		self.ensemble_size = ensemble_size
		self.base_ensemble = base_ensemble
		self.weights = None
		self.base_classifiers = []
		self.subfeatures_list = None
		
	def fit(self, train_X, train_y):
		self.fit_base_classifiers(train_X, train_y)
		self.compute_weights(train_X, train_y)
		
		
	def fit_base_classifiers(self, train_X, train_y):
		if self.base_ensemble == "cerp":
			self.fit_cerp(train_X, train_y)
		
		else:
			# set the option of number of features at each splitting node
			if self.base_ensemble == "rf":
				max_features = "sqrt"
			elif self.base_ensemble == "bagging":
				max_features = None
			else:
				raise ValueError("not a valid ensemble type")
		
		    # train base classifiers
			for i in range(self.ensemble_size):
				
				# bootstrap sampling from training set
				idxes = np.random.choice(np.arange(len(train_X)), size=len(train_X), replace=True)
				bootstrap_X = train_X[idxes]
				bootstrap_y = train_y[idxes]
				
				# train base tree
				tree = DecisionTreeClassifier(random_state=self.random_state, max_features=max_features)
				tree.fit(bootstrap_X, bootstrap_y)
				self.base_classifiers.append(tree)
	
	def fit_cerp(self, train_X, train_y):
		
		# number of features
		n_cols = train_X.shape[1]
		
		# indexes of features
		col_idxes = np.arange(n_cols)
		# randomly shuffle the indexes of features
		np.random.shuffle(col_idxes)
			
		# number of features for each base classifier
		n_cols_each = n_cols // self.ensemble_size
		
		# a list for storing index of subspaces of each base classifiers
		# the size of the list is the ensemble size
		self.subfeatures_list = []
			
		# divide the indexes of features into parts and append them to the above list
		for i in range(self.ensemble_size):
			self.subfeatures_list.append(col_idxes[i*n_cols_each:(i+1)*n_cols_each])
			
		# for indexes left, randomly assign them to some subfeatures 
		for idx in col_idxes[self.ensemble_size*n_cols_each:]:
			i = np.random.randint(self.ensemble_size)
			self.subfeatures_list[i] = np.append(self.subfeatures_list[i], idx)
		
		# train base classifiers
		for i in range(self.ensemble_size):
			# extract sub features' index from the list
			sub_features = self.subfeatures_list[i]
			
			# extract sub training set by subspace of the features
			sub_X = train_X[:, sub_features]
				
			tree = DecisionTreeClassifier()
			tree.fit(sub_X, train_y)
				
			self.base_classifiers.append(tree)
			
			
	
	def compute_weights(self, train_X, train_y):
		# performance matrix:
		#X = self.performance_matrix(train_X, train_y)
		pass
	
	
	def predict(self, new_X, return_type="label"):
		pass
	
	