## file for classifier code 
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from itertools import combinations
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.decomposition import PCA
outputs = ["no movement", "clamp close", "clamp open", "wrist left", "wrist right", "arm out", "arm in", "arm up", "arm_down"] 

def pca_transform(data,n_comp):
    pca = PCA(n_components = n_comp)
    minimized_data = pca.fit_transform(data)
    return minimized_data

# taking any six + rest trial - full list of trials and generates combination of six 
def compare_task_combinations(combined_features, labels,static_rest, trial_order):
    all_combinations = list(combinations(trial_order, 6))
    for combo in all_combinations:
        locations = [index for index, value in enumerate(labels) if value in combo]
        
        svm_list = []
        knn_list = []

        #add static data 
        temp_features = np.concatenate((combined_features[locations],static_rest[0:23,:]))
        temp_features = pca_transform(temp_features, 5)
        temp_labels = np.concatenate((labels[locations],['rest']*23))
        
        print('SVM Accuracy:'+str(train_classifier(temp_features, temp_labels, 'SVM')))
        print('KNN Accuracy:'+str(train_classifier(temp_features, temp_labels, 'KNN')))
        import pdb; pdb.set_trace()
    return locations
def determine_feature_importance(data,labels):
    
    return
def train_classifier(data, labels,model):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state = 0) 
    if model == 'SVM':
        svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
        svm_predictions = svm_model_linear.predict(X_test)  
        accuracy = svm_model_linear.score(X_test, y_test)
    if model == 'KNN':
        knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train) 
        accuracy = knn.score(X_test, y_test)
    if model == 'logistic':
        logistic = LogisticRegression(penalty='l2', C=1.0, multi_class='multinomial', solver='lbfgs', max_iter=1000)
        logistic.fit(X_train, y_train)
        y_pred = logistic.predict(X_test)
        accuracy = logistic.score(y_test, y_pred)
        
    return accuracy
def classify_gesture():
    i = random.randint(0, 8)

    return outputs[i]
