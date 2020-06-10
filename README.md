### bank-marketing
Classification problem to predict potential subscribers of a term deposit scheme for a Portuguese Bank


# SVM
## Introduction
Support vector machines offer a way for binary classification by trying to separate classes using a hyperplane in feature space. Although it is difficult to find such a hyperplane for most datasets, SVM achieves this by using 2 methods :
a)	Using a soft margin : In this we allow some points (which maybe outliers for the class) to be included on wrong side of the margin and tuning the hyperparameter C which is the allowable cost for misclassification. Tuning is done via k-fold cross validation. 
b)	Employing the kernel trick : Data sometimes might be no-separable in the original feature space but when SVM uses kernel trick to enlarge the feature space, a hyperplane maybe found to separate the classes.
## Prerequisites
After the general EDA and data cleaning tasks, SVM requires some preprocessing to be done to the data to construct a model.
1.	Irrelevant and Redundant Variables are handled by SVM hence we have not done any feature selection
2.	Missing rows are imputed in previous steps.
3.	Scaling is done by the SVM algorithm itself.
4.	One hot encoding is done for categorical variables.
5.	Data is divided --  80% training set, 20% to testing set
6.	Data is balanced using SMOTE, this is done to improve performance i.e. detecting the minority class 1 (which corresponds to “Yes”). The parameters perc.over and perc.under drives the oversampling of minority class & under sampling of the majority class. Perc.under argument controls the fraction of cases of the majority class that will be randomly selected for the final "balanced" data set.

## SVM Process:
1.	As we have about 49 columns after creation of dummy values, to visualize the data in 2 d space we have used PCA. As we can see from the plot below a lot of observations corresponding to “No” (0) is clustered towards the right side of the plot and “Yes” on the left but there are a huge number of observations which are not separable in 2-D space. Selecting an appropriate hyperplane seems difficult in 2-D space. 
 
2.	Kernel selection : We use SVM function from the package e1071 and fit the balanced training data to different types of kernels – Linear, Radial, Polynomial and Sigmoid and evaluate performance. Accuracy measures for the different kernels are :



Kernel	Accuracy
Linear 	0.8873249
Radial	0.9175647
Polynomial	0.9284079
Sigmoid	0.8171471





3.	As the best performance is on polynomial kernel, we perform hyperparameter tuning of the C and degree parameters using 10 fold cross validation and grid search. Results are shown below:
 
As we can see the best model is the one with degree 5 and cost 8.
4.	We use this model to evaluate the performance on complete train data set and we get very high values for accuracy, kappa , Recall and Precision. It almost looks to good to be true :
 5.	We use this model to check its performance on test dataset, as we can see accuracy and recall reduce to some extent but the value for kappa , precision and F1 have drastically fallen down, showing that the model has some degree of overfitting:
 6.	We evaluate the test performance on the non-cross validated polynomial SVM , which was giving the best performance earlier. Recall has increased to 87.16% and the kappa value has increased indicating moderate agreement.


## Model Performance :

Performance on cross validated model on test data (degree=5,c=8) :
 
Performance on original Polynomial SVM model on test data (degree=3,c=1) :
 
The latter has a higher AUC on test data; hence it is the final model:
 



## Final SVM Model Parameter and Performance on Complete Dataset
KERNEL:	POLYNOMIAL
DEGREE:	3
COST:	1
TRAIN ACCURACY:	92.84%
TRAIN KAPPA:	85.68%
TRAIN PRECISION:	90.84%
TRAIN RECALL:	93.01%
TEST ACCURACY:	85.71%
TEST KAPPA:	50.4%
TEST PRECISION:	43.33%
TEST RECALL:	87.1%
TEST F1:	57.85%
TEST AUC:	86.3%
 

## Conclusion
Our goal is to identify people who will subscribe to the term deposit i.e. we are aiming for a high recall. The final model has a high True positive rate (TPR) of 87.16%  and low false positive rate (FPR) of 14.5%. The value for precision is low due to the fact the positive class(1 or “Yes”) is in minority and even a small number of  false positives overwhelm the true positive and skew the precision calculation. Hence as evidenced by the AUC values, we can say that this model does a good job at identifying potential people who would subscribe to the term deposit.

