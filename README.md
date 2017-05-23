## WAVE under CERP, Random Forest and Bagging
### Overview
This project implements 3 weight-adjusted ensemble methods (Weight-Adjusted CERP, Weight-Adjusted Random Forest and WAVE under Bagging scheme) that apply a [Weight Adjusted Voting Algorithm for Ensembles of Classifiers (WAVE)](http://www.ams.sunysb.edu/~hahn/psfile/wave.pdf) to [CERP](http://www.ams.sunysb.edu/~hahn/psfile/papCERP.pdf), Random Forest and Bagging respectively. The implementation is written in Python using Object-Oriented Programming (OOP) mode that provides methods to fit models, make predictions and compute prediction accuracy etc. 

### Software Requirement
1. Python 3.4
2. Numpy
3. Pandas
4. Sci-kit Learn

### Files 
1. wave.py: implementation source code 
2. examples.ipynb: notebook consisting of examples that use WAVE under different base ensemble options
3. imp.txt: high-dimensional dataset used for the example of fitting Weight-Adjusted CERP. 

### Implementation
All the implementation are included in the **WAVE** class of wave.py file. 

In the initialization method of WAVE class, we can specify the base ensemble to be one of CERP, Random Forest and Bagging. We can also specify the ensemble size in the initialization. When the base ensemble is either Random Forest or Bagging, base tree classifiers are fully grown. On the other hand, base tree classifiers in Weight-Adjusted CERP are not fully grown. We can specify the minimum number of samples required to split an internal node for trees in CERP to control the complexity of each tree in WACERP when initializing the object of WAVE class. 

The WAVE class has a **fit()** method that calls internal helper methods to fit base classifiers and compute weight vector of base classifiers. The **predict()** method take data instances as input and return predictions. The **accuracy()** method compute accuracy for input testing instances. The WAVE class also provides **get_weights()** and **get_base_classifiers()** methods that return weight vector and a list of base classifiers respectively. 

### Examples
The jupyter notebook file **examples.ipynb** includes examples of how to call methods in WAVE class to fit these weight-adjusted ensemble methods in practice. In the examples, we show how to fit model, make predictions, compute accuracy, look into weights and base classifiers by calling methos of WAVE class. Two datasets are used in the examples. The imprinting data is a high-dimensional data set for the example of Weight-Adjusted CERP fitting. The Breast Cancer data is used for both Weight-Adjusted Random Forest and WAVE under Bagging fitting. In order to run the notebook, you need to install jupyter notebook first.
1. Fit Weight-Adjusted CERP on Imprinting Data
2. Fit Weight-Adjusted Random Forest on Breast Cancer Data
3. Fit WAVE under Bagging scheme on Breast Cancer Data
