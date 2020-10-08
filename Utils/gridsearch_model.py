import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

def grid(X, y, verbose=None, scoring=None):
    """
    for regression: 
        option 1 = LinearRegression
        option 2 = PolynomialFeatures
        option 3 = SVM - SVR
        option 4 = RandomForestRegressor
        option 5 = LogisticRegression
        option 6 = KNeighborsClassifier
        option 7 = svm - SVC
        option 8 = RandomForestClassifier()
        option 9 = XGBClassifier()
        option 10 = Perceptron
        option 11 = LinearSVC
        option 12 = SGDClassifier
        option 13 = DecisionTreeClassifier
    option to change the scoring. example: scoring='neg_root_mean_squared_error'
    """
    if verbose:
        verbose = 10
    option = int(input("Which model do you want to use? 1 = LinearRegression, 2 = PolynomialFeatures, 3 = SVM - SVR, 4 =  RandomForestRegressor, 5 = LogisticRegression, 6 = KNeighborsClassifier,  7 = svm - SVC, 8 = RandomForestClassifier, 9 = XGBClassifier"))
    

    if option == 1:
        new_model = LinearRegression()

        param_grid = [
        {'fit_intercept' : [True, False],
            'normalize' : [True, False],
            'copy_X' : [True, False]
        }
        ]
        clf = GridSearchCV(new_model, param_grid=param_grid, cv=3, n_jobs=-1)

        best_clf = clf.fit(X, y)
        return best_clf.best_estimator_

    elif option == 2:
        
        new_model = PolynomialFeatures()
        X_poly = new_model.fit_transform(X, y)

        param_grid = [
        {'degree' : np.logspace(2, 4, 6, 8),
            'interaction_only' :[True, False],
            'include_bias' : [True, False]}
        ]

        clf = GridSearchCV(new_model, param_grid=param_grid, cv=3, n_jobs=-1)
        best_clf = clf.fit(X_poly, y)
        return best_clf.best_estimator_

    elif option == 3:
        
        new_model = SVR()

        param_grid = [
        {'kernel' : ['linear', 'poly', 'rbf'],
            'degree' : [2, 4, 6],
             'C': [0.5, 1.0, 2.0]}
        ]

        clf = GridSearchCV(new_model, param_grid=param_grid, cv=3, verbose=10)
        best_clf = clf.fit(X, y)
        return best_clf.best_estimator_

    elif option == 4:
        
        new_model = RandomForestRegressor()

        param_grid = [
        {'n_estimators' : [44, 45, 47, 49, 50, 51, 53, 55],
            'max_depth' : [None, 1, 3],
            'max_features' : ['auto', 'sqrt', 'log2'],
             'warm_start': [True, False],
             #'criterion' : ['mse', 'mae'],
             'bootstrap' : [True, False],
             'oob_score' : [True, False]
             }
        ]

        clf = GridSearchCV(new_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=10)
        best_clf = clf.fit(X, y)
        return best_clf.best_estimator_
    
    elif option == 5:
        
        new_model = LogisticRegression()

        param_grid = [
        {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
            'C' : np.logspace(0, 4, 10),
            'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
             'warm_start': [True, False]}
        ]

        clf = GridSearchCV(new_model, param_grid=param_grid, cv=3, n_jobs=-1)
        best_clf = clf.fit(X, y)
        return best_clf.best_estimator_

    elif option == 6:
        
        new_model = KNeighborsClassifier()

        param_grid = [
        {'n_neighbors' : [7, 8, 9, 10, 11],
            'weights' : ['uniform', 'distance'],
            'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute']}
        ]

        clf = GridSearchCV(new_model, param_grid=param_grid, cv=3, n_jobs=-1)
        best_clf = clf.fit(X, y)
        return best_clf.best_estimator_
    
    elif option == 7:
        
        new_model = SVC()

        param_grid = [
        {
            'kernel' : ['linear', 'poly', 'rbf'],
            'C' : [-1, 1, 3],
            'degree': [3, 5, 8] }
        ]

        clf = GridSearchCV(new_model, param_grid=param_grid, cv=3, n_jobs=-1)
        best_clf = clf.fit(X, y)
        return best_clf.best_estimator_

    elif option == 8:
        
        new_model = RandomForestClassifier()

        param_grid = [
        {'n_estimators' : [50, 100, 200],
            'criterion' : ['gini', 'entropy'],
            'warm_start': [True, False],
            'max_features': ['auto', 'sqrt', 'log2']}
        ]

        clf = GridSearchCV(new_model, param_grid=param_grid, cv=3, n_jobs=-1)
        best_clf = clf.fit(X, y)
        return best_clf.best_estimator_

    elif option == 9:
        
        new_model = XGBClassifier()

        param_grid = [
        {'booster' : ['gbtree', 'gblinear','dart'],
            'max_delta_step':[0.1, 0, 1.],
            'max_depth': [6, 8, 10]}
        ]

        clf = GridSearchCV(new_model, param_grid=param_grid, cv=3, n_jobs=-1)
        best_clf = clf.fit(X, y)
        return best_clf.best_estimator_

    
    elif option == 10:
        
        new_model = Perceptron()

        param_grid = [
        {'penalty' : [None, 'l2','l1','elasticnet'],
            'fit_intercept':[True, False],
            'shuffle': [True, False]}
        ]

        clf = GridSearchCV(new_model, param_grid=param_grid, cv=3, n_jobs=-1)
        best_clf = clf.fit(X, y)
        return best_clf.best_estimator_

    elif option == 11:
        
        new_model = LinearSVC()

        param_grid = [
        {'penalty' : ['l1', 'l2'],
            'loss':['hinge', 'squared_hinge'],
            'C': [1, 3, 5, 8]}
        ]

        clf = GridSearchCV(new_model, param_grid=param_grid, cv=3, n_jobs=-1)
        best_clf = clf.fit(X, y)
        return best_clf.best_estimator_

    elif option == 12:
        
        new_model = SGDClassifier()

        param_grid = [
        {'penalty' : ['l2', 'l1', 'elasticnet'],
            'early_stopping':[True, False],
            'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge']}
        ]

        clf = GridSearchCV(new_model, param_grid=param_grid, cv=3, n_jobs=-1)
        best_clf = clf.fit(X, y)
        return best_clf.best_estimator_

    elif option == 13:
        
        new_model = DecisionTreeClassifier()

        param_grid = [
        {'criterion' : ['gini', 'entropy'],
            'splitter':['best', 'random'],
            'min_samples_split': [1, 2, 5, 10]}
        ]

        clf = GridSearchCV(new_model, param_grid=param_grid, cv=3, n_jobs=-1)
        best_clf = clf.fit(X, y)
        return best_clf.best_estimator_
        

   

   