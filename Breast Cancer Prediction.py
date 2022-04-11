#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt


# In[ ]:


#data.csv needs to be uploaded into the Files section on the left
data = pd.read_csv('data.csv')


# In[ ]:


data.head()


# In[ ]:


print(data.shape) #shape of data
print(data.columns) #columns of data


# In[ ]:


#Explore all the columns of data
data.info()


# #### Attribute Information: ####
# 
# 1) ID number
# 2) Diagnosis (M = malignant, B = benign)
# 3-32)
# 
# Ten real-valued features are computed for each cell nucleus:
# 
# a) radius (mean of distances from center to points on the perimeter)
# b) texture (standard deviation of gray-scale values)
# c) perimeter
# d) area
# e) smoothness (local variation in radius lengths)
# f) compactness (perimeter^2 / area - 1.0)
# g) concavity (severity of concave portions of the contour)
# h) concave points (number of concave portions of the contour)
# i) symmetry
# j) fractal dimension ("coastline approximation" - 1)
# 
# The mean, standard error and "worst" or largest (mean of the three
# largest values) of these features were computed for each image,
# resulting in 30 features. For instance, field 3 is Mean Radius, field
# 13 is Radius SE, field 23 is Worst Radius.
# 
# All feature values are recoded with four significant digits.
# 
# Missing attribute values: none
# 
# Class distribution: 357 benign, 212 malignant

# In[ ]:


#Statistics of data
data.describe()


# In[ ]:


data['Unnamed: 32'].unique() #This column is all Nans, can be dropped 


# In[ ]:


#Split data into features (X) and target variable (y)
#Drop columns: Unnamed: 32 and id since aren't relevant

todrop = ['id', 'Unnamed: 32', 'diagnosis']
X = data.drop(todrop, axis = 1)
X.head()
y = data['diagnosis'].map({'M':1,'B':0}) #Convert Malignant and Benign labels to binary, Malignant = 1 and Benign = 0  


# In[ ]:


#How many Malignant and Benign datapoints in the dataset
sns.countplot(data['diagnosis'],label="Count");

benign, malignant =  y.value_counts()
print('Number of Benign:', benign)
print('Number of Malignant:', malignant)


# ### Data Visualization ###
# 

# In[ ]:


data.iloc[:,1:11].hist(figsize=(10,12), bins=20, layout=(5,2), grid=False)
plt.tight_layout()
# plot the distribution for each feature and look for skewness on the mean values


# In[ ]:


# It's obvious that features like concavity and area are largely skewed.
skewness = data.iloc[:,1:11].skew()
skewness


# In[ ]:


# normalize the data
X_norm = (X - X.mean()) / (X.std())
X_norm.head()


# In[ ]:


data = pd.concat([y,X_norm.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(8,8))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")
plt.xticks(rotation=90)


# From the above violin plot, we can see that the median of Malignant and Benign for concave points and perimeter are quite different, indicating that these would be good features for classification.

# In[ ]:


data = pd.concat([y,X_norm.iloc[:,10:20]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(8,8))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")
plt.xticks(rotation=90)


# From this violin plot above, we can observe that the area_se feature and radius_se fearure contain a quite different median of Malignant and Benign.

# In[ ]:



data = pd.concat([y,X_norm.iloc[:,20:31]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(8,8))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")
plt.xticks(rotation=90)


# In[ ]:


corr = X.corr().round(2)

# Mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set figure size
f, ax = plt.subplots(figsize=(17, 13))

# Define custom colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.tight_layout()


# All the "mean" columns are highly correlated with the "worst" columns. Therefore, we should discard all the "worst" columns. Radius, peremiter and area are highly correlated, thus we will drop columns related with peremiter and area. Compactness, concavity and concave points are highly related, thus we will drop columns related with concavity and concave points.

# In[ ]:


cols = ['radius_worst', 
        'texture_worst', 
        'perimeter_worst', 
        'area_worst', 
        'smoothness_worst', 
        'compactness_worst', 
        'concavity_worst',
        'concave points_worst', 
        'symmetry_worst', 
        'fractal_dimension_worst']
X_norm = X_norm.drop(cols, axis=1)


# In[ ]:


cols = ['perimeter_mean',
        'perimeter_se', 
        'area_mean', 
        'area_se']
X_norm = X_norm.drop(cols, axis=1)


# In[ ]:


cols = ['concavity_mean',
        'concavity_se', 
        'concave points_mean', 
        'concave points_se']
X_norm = X_norm.drop(cols, axis=1)


# In[ ]:


corr = X_norm.corr().round(2)
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
plt.tight_layout()


# Above is the new heatmap after we delete the columns that are highly correlated, and it's obvious that the correlation between the current columns are quite low. Therefore, now we can use this data set to train models.

# ## Models

# In[ ]:


from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.15, random_state=42)


# ### Logistic Regression

# In[ ]:


parameters = {'penalty':('l1', 'l2', 'elasticnet','none'), 'C':[1, 10, 100, 1000], 'solver':('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'), 'max_iter':[90,100,110]}
logistic = LogisticRegression()
clf = GridSearchCV(logistic, parameters)
clf.fit(X_train, y_train)


# In[ ]:


logistic = LogisticRegression(penalty='l2', C=1, solver='lbfgs')
logistic.fit(X_train, y_train)
sns.heatmap(confusion_matrix(y_train, logistic.predict(X_train)), annot=True, fmt='g');


# In[ ]:


# Metrics on the test set

accuracy = accuracy_score(y_test, logistic.predict(X_test))
f1 = f1_score(y_test, logistic.predict(X_test))
recall = recall_score(y_test, logistic.predict(X_test))
precision = precision_score(y_test, logistic.predict(X_test))

metrics = pd.DataFrame({'Logistic Regression':[accuracy,f1,recall,precision]}, ['Accuracy','F1 Score','Recall','Precision'])


# ### KNN

# In[ ]:


parameters={'n_neighbors':[3,5,7,9], 'weights':('uniform', 'distance'), 'algorithm':('auto', 'ball_tree', 'kd_tree', 'brute'), 'leaf_size':[25,30,35]}
knn = KNeighborsClassifier()
clf = GridSearchCV(knn, parameters)
clf.fit(X_train, y_train)


# In[ ]:


knn = KNeighborsClassifier(algorithm='auto', leaf_size=30,
                                            metric='minkowski',
                                            metric_params=None, n_jobs=None,
                                            n_neighbors=5, p=2,
                                            weights='uniform')
knn.fit(X_train,y_train)
sns.heatmap(confusion_matrix(y_train, knn.predict(X_train)),annot=True, fmt='g');


# In[ ]:


# Metrics on the test set

accuracy = accuracy_score(y_test, knn.predict(X_test))
f1 = f1_score(y_test, knn.predict(X_test))
recall = recall_score(y_test, knn.predict(X_test))
precision = precision_score(y_test, knn.predict(X_test))

metrics['KNN'] = [accuracy, f1,recall,precision]


# ### Decision Tree 

# In[ ]:


parameters = {'max_depth': range(1,5), 'max_features': range(3,6), 'criterion': ['gini','entropy']}
dt = DecisionTreeClassifier(random_state=17)
clf = GridSearchCV(dt, parameters, cv=10)
clf.fit(X_train, y_train)


# In[ ]:


#Visualize Tree
from sklearn.tree import export_graphviz
tree_graph = export_graphviz(clf.best_estimator_, class_names = ['benign', 'malignant'], feature_names = X_train.columns, filled=True, out_file='tree.dot')
get_ipython().system('dot -Tpng tree.dot -o tree.png ')


# In[ ]:


from IPython.display import Image
Image(filename = 'tree.png')


# Radius_mean is the root node and has the highest information gain, which is why it is split first, so size of the cell nucleus is significant in classifying benign or malignant. The second nodes are texture_se and fractal_dimension_mean, this can be interpreted as contrast in the photograph, so the less contrasted the picture is, the more probability that the cell nucleus is benign.

# ### Random Forest

# In[ ]:


lst = list(np.arange(50,200,10))
parameters={'criterion':('gini', 'entropy'),'max_depth':lst, 'max_features':('auto', 'sqrt', 'log2')}
rf = RandomForestClassifier()
clf = GridSearchCV(rf, parameters)
clf.fit(X_train, y_train)


# In[ ]:


rf = RandomForestClassifier(criterion='gini', max_depth=100, n_estimators=100, max_features='auto')
rf.fit(X_train, y_train)
sns.heatmap(confusion_matrix(y_train, rf.predict(X_train)),annot=True, fmt='g');


# In[ ]:


# Metrics on the test set

accuracy = accuracy_score(y_test, rf.predict(X_test))
f1 = f1_score(y_test, rf.predict(X_test))
recall = recall_score(y_test, rf.predict(X_test))
precision = precision_score(y_test, rf.predict(X_test))

metrics['Random Forest'] = [accuracy, f1,recall,precision]


# ### Neural Network

# In[ ]:


#parameters = {'hidden_layer_sizes':[(100),(100,50),(50,100),(50),(100,100)]}
parameters = {'activation':('identity', 'logistic', 'tanh', 'relu'), 'solver':('sgd', 'lbfgs', 'adam')}
nn = MLPClassifier(activation='relu',solver='adam')
clf = GridSearchCV(nn, parameters)
clf.fit(X_train, y_train)


# In[ ]:


nn = MLPClassifier(hidden_layer_sizes=(100,), activation='relu',solver='adam')
nn.fit(X_train,y_train)
sns.heatmap(confusion_matrix(y_train, nn.predict(X_train)),annot=True, fmt='g');


# In[ ]:


# Metrics on the test set

accuracy = accuracy_score(y_test, nn.predict(X_test))
f1 = f1_score(y_test, nn.predict(X_test))
recall = recall_score(y_test, nn.predict(X_test))
precision = precision_score(y_test, nn.predict(X_test))

metrics['Neural Network'] = [accuracy, f1,recall,precision]


# ### SVM

# In[ ]:


parameters={'kernel':('linear', 'poly', 'rbf', 'sigmoid'), 'gamma':('scale', 'auto'), 'C':[1,10,100,1000]}
svm = SVC()
clf = GridSearchCV(svm, parameters)
clf.fit(X_train, y_train)


# In[ ]:


svm = SVC(C=1, gamma='scale', kernel='rbf', probability=True)
svm.fit(X_train,y_train)
sns.heatmap(confusion_matrix(y_train, svm.predict(X_train)),annot=True, fmt='g');


# In[ ]:


# Metrics on the test set

accuracy = accuracy_score(y_test, svm.predict(X_test))
f1 = f1_score(y_test, svm.predict(X_test))
recall = recall_score(y_test, svm.predict(X_test))
precision = precision_score(y_test, svm.predict(X_test))

metrics['SVM'] = [accuracy, f1,recall,precision]


# ### AdaBoost

# In[ ]:


parameters={'algorithm':('SAMME', 'SAMME.R'), 'n_estimators':[45,50,55], 'learning_rate':[.05,.1,.5,1]}
ab = AdaBoostClassifier()
clf = GridSearchCV(ab, parameters)
clf.fit(X_train, y_train)


# In[ ]:


ab = AdaBoostClassifier(algorithm='SAMME.R', learning_rate=1.0, n_estimators=50)
ab.fit(X_train,y_train)
sns.heatmap(confusion_matrix(y_train, ab.predict(X_train)), annot=True, fmt='g');


# In[ ]:


# Metrics on the test set

accuracy = accuracy_score(y_test, ab.predict(X_test))
f1 = f1_score(y_test, ab.predict(X_test))
recall = recall_score(y_test, ab.predict(X_test))
precision = precision_score(y_test, ab.predict(X_test))

metrics['AdaBoost'] = [accuracy, f1,recall,precision]


# ### Results

# In[ ]:


metrics


# In[ ]:


sns.barplot(metrics['Logistic Regression'].index, metrics['Logistic Regression'].values)


# In[ ]:


plt.bar(metrics['Logistic Regression'].index, metrics['Logistic Regression'].values,color=['black', 'red', 'green', 'blue'])
plt.ylim(.74,1.01)
plt.title("Logistic Regression")
plt.show();
plt.bar(metrics['KNN'].index, metrics['KNN'].values,color=['black', 'red', 'green', 'blue'])
plt.ylim(.74,1.01)
plt.title("KNN")
plt.show();
plt.bar(metrics['Random Forest'].index, metrics['Random Forest'].values,color=['black', 'red', 'green', 'blue'])
plt.ylim(.74,1.01)
plt.title("Random Forest")
plt.show();
plt.bar(metrics['Neural Network'].index, metrics['Neural Network'].values,color=['black', 'red', 'green', 'blue'])
plt.ylim(.74,1.01)
plt.title("Neural Network")
plt.show();
plt.bar(metrics['SVM'].index, metrics['SVM'].values,color=['black', 'red', 'green', 'blue'])
plt.ylim(.74,1.01)
plt.title("SVM")
plt.show();
plt.bar(metrics['AdaBoost'].index, metrics['AdaBoost'].values,color=['black', 'red', 'green', 'blue'])
plt.ylim(.74,1.01)
plt.title('AdaBoost')
plt.show();


# After training through GridSearch corss validation and evaluating the models on the test set, it appears that both logisitcs regression and the neural network performed the best, with a F1 score of 90.63%.

# ## Clustering ##

# In[ ]:


from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X) 


# ### Principal Component Analysis ###

# In[ ]:


from sklearn.decomposition import PCA # Principal Component Analysis module


# In[ ]:


#Let's graph how many components is the ideal amount. 
# This can be determined by seeing how many components can explain 90% of the initial data dispersion (quantified by the explained_variance_ratio)
decomp = PCA().fit(X_scaled)

plt.figure(figsize=(10,7))
plt.plot(np.cumsum(decomp.explained_variance_ratio_), color='k', lw=2)
plt.xlabel('Number of components')
plt.ylabel('Explained variance (total)')
plt.xlim(0, 29)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.axvline(6, c='b')
plt.axhline(0.91, c='b')
plt.plot(6, 0.91, 'ro')
plt.show();


# We see that 6 components capture 91% of the data variance.

# In[ ]:


pca = PCA(n_components=6)
pca_results = pca.fit_transform(X_scaled) 


# In[ ]:


feat_imp = pd.DataFrame(pca.components_)

feat_imp.columns = X.columns
feat_imp


# In[ ]:


plt.figure(figsize=(10,7))
sns.barplot(x = feat_imp.columns, y = abs(feat_imp.iloc[0,:])).set_title('Feature Importance: PC1')
plt.ylabel('Mean of Eigenvector Values across PC1')
plt.xlabel('Features')
plt.xticks(rotation=80);


# In[ ]:


plt.figure(figsize=(10,7))
sns.barplot(x = feat_imp.columns, y = abs(feat_imp.mean(axis=0))).set_title('Feature Importance: PC1-PC6')
plt.ylabel('Mean of Eigenvector Values across all 6 Principal Components')
plt.xlabel('Features')
plt.xticks(rotation=80);


# It looks like the standard error features, specifically from compactness to fractal dimension, have the highest mean of eigen vector values across the six principal components, the higher magnitude of eigenvector values indicateds greater feature importance.

# ### TSNE ###

# In[ ]:


from sklearn.manifold import TSNE # TSNE module

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=2000)
tsne_results = tsne.fit_transform(X_scaled)


# In[ ]:


#Visualization 
plt.figure(figsize = (16,11))
plt.subplot(121)
plt.scatter(pca_results[:,0],pca_results[:,1], c = y,  alpha=0.35) #pca_results[:,2]
plt.colorbar()
plt.title('PCA Scatter Plot')
plt.subplot(122)
plt.scatter(tsne_results[:,0],tsne_results[:,1],  c = y, cmap = "coolwarm", edgecolor = "None", alpha=0.35)
plt.colorbar()
plt.title('TSNE Scatter Plot')
plt.show()


# In[ ]:


#PCA and visualization on the feature engineered dataset (X_norm?)

pca2 = PCA(n_components=6)
pca2_results = pca2.fit_transform(X_norm) 


# In[ ]:


def plotvectors(score,coeff,labels=None):
    #Author of function: Serafeim Loukas, serafeim.loukas@epfl.ch
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley, c = y)
    for i in range(5):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
 
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')

plt.figure(figsize = (16,11))

plt.subplot(121)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Before Feature Engineering: More colinear features')
plt.grid()
plotvectors(pca_results[:,0:2], np.transpose(pca.components_[0:2, :]))

plt.subplot(122)
plt.title('After Feature Engineering: More linearly independent features')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid()
plotvectors(pca2_results[:,0:2], np.transpose(pca2.components_[0:2, :]))

plt.show()


# In[ ]:


#Feature Importance Analysis for Feature Engineered Dataset

feat_imp2 = pd.DataFrame(pca2.components_)

feat_imp2.columns = X_norm.columns
feat_imp2


# In[ ]:


plt.figure(figsize=(10,7))
sns.barplot(x = feat_imp2.columns, y = abs(feat_imp2.iloc[0,:])).set_title('Feature Importance: PC1')
plt.ylabel('Mean of Eigenvector Values across PC1')
plt.xlabel('Features')
plt.xticks(rotation=80);


# In[ ]:


plt.figure(figsize=(10,7))
sns.barplot(x = feat_imp2.columns, y = abs(feat_imp2.mean(axis=0))).set_title('Feature Importance: PC1-PC6')
plt.ylabel('Mean of Eigenvector Values across all 6 Principal Components')
plt.xlabel('Features')
plt.xticks(rotation=80);


# It is clear that radius_se is the most important feature across all 6 principal components in the cleaned and reduced dataset (after feature engineering).  
