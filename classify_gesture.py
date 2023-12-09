## file for classifier code 
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
outputs = ["no movement", "clamp close", "clamp open", "wrist left", "wrist right", "arm out", "arm in", "arm up", "arm_down"] 
import pickle 


def pca_transform(data,n_comp):
    pca = PCA(n_components = n_comp)
    minimized_data = pca.fit_transform(data)
    return minimized_data

# taking any six + rest trial - full list of trials and generates combination of six 
def compare_task_combinations(combined_features, labels,static_rest, trial_order,path):
    all_combinations = list(combinations(trial_order, 6))
    
    for combo in all_combinations:
        locations = [index for index, value in enumerate(labels) if value in combo]
        
        svm_list = []
        knn_list = []


        #add static data 
        temp_features = np.concatenate((combined_features[locations],static_rest[0:23,:]))
      #  temp_features = pca_transform(temp_features, 5)
        temp_labels = np.concatenate((labels[locations],['rest']*23))

        svm_accuracy, svm_model, score_svm = train_classifier(temp_features, temp_labels, 'SVM')
        knn_accuracy, knn_model, score_knn = train_classifier(temp_features, temp_labels, 'KNN')
        logistic_accuracy, logistic_model, score_log = train_classifier(temp_features, temp_labels, 'logistic')

        if (combo == ('hand_fist', 'wrist_down', 'two_finger_pinch', 'wrist_right', 'wrist_left', 'hand_open')):
            pickle.dump(svm_model, open('model.pkl', 'wb'))

        print("Combination: ", combo)
        print('SVM Accuracy:'+str(svm_accuracy))
        print('SVM Score:'+str(np.mean(score_svm)))
       
        print('\nKNN Accuracy:'+str(knn_accuracy))
        print('KNN Score:'+str(np.mean(score_knn)))

        print('\nlogistic Accuracy:'+str(logistic_accuracy))
        print('logistic Score:'+str(np.mean(score_log))+'\n')
      #  import pdb; pdb.set_trace()
    return locations

def determine_feature_importance(data,labels):
    
    return

def train_classifier(data, labels,model):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state = 0) 
    if model == 'SVM':
       # print(np.shape(X_train))
        svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
        model = svm_model_linear
        svm_predictions = svm_model_linear.predict(X_test)  
        accuracy = svm_model_linear.score(X_test, y_test)

        # kFold validation 
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        score = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')

    if model == 'KNN':
        knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train) 
        model = knn
        accuracy = knn.score(X_test, y_test)

        # kFold validation 
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        score = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')

    if model == 'logistic':
        label_encoder = LabelEncoder() #initializing label encoder 

        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.fit_transform(y_test)

      #  y_train = y_train.reshape(-1,1)

      #  X_train = X_train.reshape(-1, 5)
     #   X_test = X_test.reshape(-1, 5)

        logistic = LogisticRegression(penalty='l2', C=1.0, multi_class='multinomial', solver='lbfgs', max_iter=10000)
        model = logistic 
        logistic.fit(X_train, y_train.ravel())
        y_pred = logistic.predict(X_test)
        accuracy = accuracy_score(y_test.ravel(), y_pred.ravel())

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        score = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')

        
    return accuracy, model,score

def cross_validate(model,metric):
    num_folds = 5  # Replace with your chosen number of folds
    kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring=metric)



