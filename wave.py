import numpy as np
from sklearn.tree import DecisionTreeClassifier

class WAVE(object):
	"""
	Weight-Adjusted Voting algorithm for Ensembles of Classifiers (WAVE)
	
	This class provides methods to implement WAVE ensemble under different options of aggregation schema. 
	Base Ensemble Option: 1.CERP, 2.Random Forest, 3.Bagging
	Base Classifier in the Ensemble: Decision Tree
	"""
	
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
		    train_X: 2-d numpy array, size=(n, p)
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
		    train_X: 2-d numpy array, size=(n, p)
		    train_Y: 1-d numpy array, size=(n, )
			
		Return: None
		Update: self.base_classifiers
		"""
		# set list of base classifiers to be empty
		self.base_classifiers = []
		
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
		    train_X: 2-d numpy array, size=(n, p)
		    train_Y: 1-d numpy array, size=(n, )
		    min_samples_split: The minimum number of samples required to split an internal node	for trees in CERP.
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
		Compute Weights for Base Classifiers
		Args:
		    train_X: 2-d numpy array, size=(n, p)
		    train_Y: 1-d numpy array, size=(n, )
		Return: None
		Update: self.weights
		"""
		# n is the number of instances in training set
		n = len(train_X)
		# k is the number of base classifiers in the ensemble
		k = self.ensemble_size
		
		# call a helper function to compute performance matrix X
		# X has the shape (n, k)
		X = self.performance_matrix(train_X, train_y)
		
		# Define two matrices consisting of 1s
		J_nk = np.ones((n, k))
		J_kk = np.ones((k, k))
		
		# Define an identity matrix
		identity_k = np.identity(k)
		
		# Compute the matrix T 
		T = X.T.dot((J_nk-X)).dot((J_kk - identity_k))
		
		# Find eigenvalues and eigenvectors of T 
		eig_values, eig_vectors = np.linalg.eig(T)[0].real, np.linalg.eig(T)[1].real
		
		# Find r domain eigenvalues 
		max_eig_value = eig_values.max()
		r = 0
		idxes_max_eig = []
		for i in range(len(eig_values)):
			if eig_values[i] == max_eig_value:
				r += 1
				idxes_max_eig.append(i)
		
		# compute matrix sigma
		sigma = np.zeros((k, k))
		for i in range(r):
			u = eig_vectors[:, idxes_max_eig[i]]
			u = u.reshape((k, 1))
			sigma += u.dot(u.T)
		
		# Define a vector of 1s
		k_1 = np.ones((k, 1))
		
		# Compute the weight vector and set it as self.weights
		self.weights = (sigma.dot(k_1)) / k_1.T.dot(sigma).dot(k_1)
			
	def performance_matrix(self, train_X, train_y):
		"""
		helper function to compute performance matrix
		Args:
		    train_X: 2-d numpy array, size=(n, p)
		    train_Y: 1-d numpy array, size=(n, )
			
		Return: performance matrix X
		        shape of X: (n, k), where k is the ensemble size
		        each element of X is either 1 or 0
		"""
		
		# Initialize X as the predictions of the training set by the first base classifier
		# After initialization, shape of X is (n, ), each element of X is either True or False
		if self.base_ensemble == "cerp":
			X = self.base_classifiers[0].predict(train_X[:, self.subfeatures_list[0]]) == train_y
		else:
			X = self.base_classifiers[0].predict(train_X) == train_y
			
		# trainsform elements of X from boolean to int: True=>1, False=>0
		X = X.astype(int)
		
		# For each of the other base classifiers, make predictions of training set
		for i in range(1, self.ensemble_size):
			#print (i)
			if self.base_ensemble == "cerp":
				column_i = self.base_classifiers[i].predict(train_X[:, self.subfeatures_list[i]]) == train_y
			else:
				column_i = self.base_classifiers[i].predict(train_X) == train_y
			column_i = column_i.astype(int)
			
			# attach predictions of each base classifier to X as a new column
			X = np.column_stack((X, column_i))
		
		return X
	
	def get_weights(self):
		"""
		Return: Weight Vector of Base Classifiers
		        shape: 2-d array (k, 1), where k is the ensemble size
		"""
		return self.weights
	
	def get_base_classifiers(self):
		"""
		Return: a list of base classifiers
		"""
		return self.base_classifiers
	
	def predict(self, new_X, return_type="label"):
		"""
		Args:
		    new_X: new instance(s) for making prediction
		           shape of new_X: either 1-d array (one instance), (p, )
				                     or   2-d array (multiple instances), (n, p)
		    return_type: str, either "label" or "prob"
		
		Return:
		    a list consisting of of predictions
		        if input return_type is "label": each prediction is the predicted label
		        if input return_type is "prob" : each prediction is a dictionary, where
				                                 key is possible label, and
												 value is corresponding predicted probablity							 
		"""
		
		#Initialize the predictions as an empty list
		predictions = []
			
		# check the shape of new_X
		# if the shape is 1-d array (k,), then reshape it to 2-d array (1, k)
		# reshape is necessary for making prediction
		if len(new_X.shape) == 1:
			new_X = new_X.reshape((1, -1))
			
		for idx in range(len(new_X)):
			new_instance = new_X[[idx]]
			
			# Initialize the prediction dictionary
		    # key is each possible label
		    # value is the predicted probability of the label, initialized as 0
			pred_dict = {}
			for label in self.class_labels:
			    pred_dict[label] = 0
			
		    # making predictions, update pred_dict
			if self.base_ensemble == "cerp":
			    # for CERP, each base classifier makes predictions only use subset of features of new_instance
			    for i in range(self.ensemble_size):
				    # extract subfeatures of new_X for making prediction by current base classifier
				    features_idxes = self.subfeatures_list[i]
				    sub_X = new_instance[:, features_idxes]
				    pred_label = self.base_classifiers[i].predict(sub_X)[0] # a scalar, not an array
				    weit = self.weights[i][0]  # a scalar, not an array
				    pred_dict[pred_label] += weit
			else:
			    # in the cases where base ensemble is either Bagging or Random Forest
			    for i in range(self.ensemble_size):
				    pred_label = set.base_classifiers[i].predict(new_instance)[0]  # a scalar, not an array
				    weit = self.weights[i][0]  # a scalar, not an array
				    pred_dict[pred_label] += weit
			
		    # if the return_type is chosen to be "label", find the label that has the highest weight
			if return_type == "label":
			    prob = 0
			    ans_label = None
			    for label in pred_dict.keys():
				    if pred_dict[label] > prob:
					    prob = pred_dict[label]
					    ans_label = label
			    predictions.append(ans_label)
			else:
			    # in the case here return_type is chosen to be "prob", just return the pred_dict
			    predictions.append(pred_dict)
				
		return predictions
		
	
	def accuracy(self, test_X, test_y):
		"""
		Compute the prediction accuracy on the given test set
		
		Args:
		    test_X: either a 1-d array or a 2-d array
		    test_y: 1-d array
			
		Return:
		    float: accuracy on test set
		"""
		predictions = self.predict(test_X)
		return np.mean(predictions == test_y)

	
	
	