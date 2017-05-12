import numpy as np
from sklearn.tree import DecisionTreeClassifier

class WAVE(object):
	
	def __init__(self, ensemble_size, base_ensemble="rf", min_samples_split_cerp=5):
		"""
		Args:
			ensemble_size: int, number of base classifiers
			base_ensemble: option of base ensemble type, one of the following 3:
						   "rf":      Random Forest
						   "bagging": Bagging
						   "cerp":    CERP
			min_samples_split_cerp: int, >= 2, default = 5,
			               The minimum number of samples required to split an internal node	for trees in CERP
						   This argument controlls the complexity of base trees in CERP
		"""
		self.ensemble_size = ensemble_size
		self.base_ensemble = base_ensemble
		self.min_samples_split_cerp = min_samples_split_cerp
		
		# weight vector for base classifiers
		self.weights = None
		
		# list for storing base classifiers
		self.base_classifiers = []
		
		# list of subfeatures's column indexes of CERP
		# each element is a 1-d numpy array consisting of column indexes of training features
		# this list is for predicting use of CERP
		self.subfeatures_list = None
		
		# list of class labels
		self.class_labels = None
		
	def fit(self, train_X, train_y):
		"""
		Fit the WAVE Ensemble consisting of base classifiers and corresponding weights
		Args:
		    train_X: 2-d numpy array, size=(n, k)
			train_Y: 1-d numpy array, size=(n, )
		
		Return: None
		Update: self.class_labels
		        self.base_classifiers
		        self.weights 
		"""
		# compute class labels 
		self.class_labels = np.unique(train_y)
		
		# fit base classifiers of the ensemble using helper function
		self.fit_base_classifiers(train_X, train_y)
		
		# compute weight vector using helper function
		self.compute_weights(train_X, train_y)
		
		
	
	def fit_base_classifiers(self, train_X, train_y):
		"""
		Fit Base Classifiers, a helper function used in fit() method
		Args:
		    train_X: 2-d numpy array, size=(n, k)
			train_Y: 1-d numpy array, size=(n, )
			
		Return: None
		Update: self.base_classifiers
		"""
		# if the base ensemble is CERP, call fit_cerp helper function
		if self.base_ensemble == "cerp":
			self.fit_cerp(train_X, train_y, self.min_samples_split_cerp)
		
		# if base ensemble is Random Forest or Bagging
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
				tree = DecisionTreeClassifier(max_features=max_features)
				tree.fit(bootstrap_X, bootstrap_y)
				
				# append the base tree to the list of base classifiers
				self.base_classifiers.append(tree)
	
	def fit_cerp(self, train_X, train_y, min_samples_split):
		"""
		Fit CERP ensemble, a helper function used in fit_base_classifiers() method
		Args:
		    train_X: 2-d numpy array, size=(n, k)
			train_Y: 1-d numpy array, size=(n, )
			min_samples_split: The minimum number of samples required to split an internal node	for trees in CERP
						       This argument controlls the complexity of base trees in CERP
		Return: None
		Update: self.base_classifiers
		"""
		
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
			# extract sub features' indexes from the list
			sub_features = self.subfeatures_list[i]
			
			# extract sub training set by subspace of the features
			sub_X = train_X[:, sub_features]
				
			tree = DecisionTreeClassifier(min_samples_split = min_samples_split)
			tree.fit(sub_X, train_y)
			
			# update self.base_classifiers by appending the base tree
			self.base_classifiers.append(tree)
			
			
	
	def compute_weights(self, train_X, train_y):
		"""
		Comput Weights for Base Classifiers
		
		"""
		# performance matrix:
		X = self.performance_matrix(train_X, train_y)
		
		n = len(train_X)
		k = self.ensemble_size
		
		J_nk = np.ones((n, k))
		J_kk = np.ones((k, k))
		
		identity_k = np.identity(k)
		
		T = X.T.dot((J_nk-X)).dot((J_kk - identity_k))
		
		eig_values, eig_vectors = np.linalg.eig(T)[0].real, np.linalg.eig(T)[1].real
		
		max_eig_value = eig_values.max()
		r = 0
		idxes_max_eig = []
		for i in range(len(eig_values)):
			if eig_values[i] == max_eig_value:
				r += 1
				idxes_max_eig.append(i)
				
		sigma = np.zeros((k, k))
		
		for i in range(r):
			u = eig_vectors[:, idxes_max_eig[i]]
			u = u.reshape((k, 1))
			sigma += u.dot(u.T)
			
		k_1 = np.ones((k, 1))
		
		self.weights = (sigma.dot(k_1)) / k_1.T.dot(sigma).dot(k_1)
			
	
	def performance_matrix(self, train_X, train_y):
		if self.base_ensemble == "cerp":
			X = self.base_classifiers[0].predict(train_X[:, self.subfeatures_list[0]]) == train_y
		else:
			X = self.base_classifiers[0].predict(train_X) == train_y
		X = X.astype(int)
		for i in range(1, self.ensemble_size):
			if self.base_ensemble == "cerp":
				column_i = self.base_classifiers[i].predict(train_X[:, self.subfeatures_list[i]]) == train_y
			else:
				column_i = self.base_classifiers[i].predict(train_X) == train_y
			column_i = column_i.astype(int)
			X = np.column_stack((X, column_i))
		
		return X
	
	
	def get_weights(self):
		return self.weights
	
	def get_base_classifiers(self):
		return self.base_classifiers
	
	def predict(self, new_X, return_type="label"):
		pred_dict = {}
		for label in self.class_labels:
			pred_dict[label] = 0
			
		if len(new_X.shape) == 1:
			new_X = new_X.reshape((1, -1))
			
		if self.base_ensemble == "cerp":
			for i in range(self.ensemble_size):
				features_idxes = self.subfeatures_list[i]
				sub_X = new_X[:, features_idxes]
				pred_label = self.base_classifiers[i].predict(sub_X)[0]
				weit = self.weights[i]
				pred_dict[pred_label] += weit
		else:
			for i in range(self.ensemble_size):
				pred_label = set.base_classifiers[i].predict(new_X)[0]
				pred_dict[pred_label] += weit
			
		if return_type == "label":
			prob = 0
			ans_label = None
			for label in pred_dict.keys():
				if pred_dict[label] > prob:
					prob = pred_dict[label]
					ans_label = label
			return label
		else:
			return pred_dict
		

	
	
	