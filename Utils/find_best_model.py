import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np





def best_model(X_train, y_train):
    """
    Function that return a plot showing the 'neg_root_mean_squared_error' of each model. 
    According to the kind of problem (regression/classification) the model plots the corresponding graphs 
    #TODO check if more models need to be added.  

    """
    # prepare configuration for cross validation test harness
    seed = 7
    
    choice = int(input("what kind of models do you want to try? Put '1' for regression or put '2' for classification"))
    # prepare models for regression
    if choice == 1:
        models = []
        models.append(('Linear Regression', LinearRegression()))
        models.append(('Polynominal Linear Regression', LinearRegression()))
        models.append(('Support Vector Regression', SVR(kernel = 'rbf')))
        models.append(('Decision Tree Regressor', DecisionTreeRegressor()))
        models.append(('Random Forest Regressor', RandomForestRegressor(n_estimators = 10)))


        # evaluate each model in turn
        results = []
        names = []
        scoring = 'neg_root_mean_squared_error'
        for name, model in models:
            kfold = model_selection.KFold(n_splits=4, random_state=seed)
            if name == 'PR':
                poly_reg = PolynomialFeatures(degree = 4)
                X_poly = poly_reg.fit_transform(X_train)
                cv_results = model_selection.cross_val_score(model, X_poly, y_train.ravel(), cv=kfold, scoring=scoring)
            else:
                cv_results = model_selection.cross_val_score(model, X_train, y_train.ravel(), cv=kfold, scoring=scoring)

            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

        # boxplot algorithm comparison
        fig = plt.figure()
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        plt.xticks(rotation=90)
        plt.show()


    elif choice == 2:
        models = []
        models.append(('KNeighborsClassifier', KNeighborsClassifier()))
        models.append(('Logistic Regression', LogisticRegression()))
        models.append(('Support Vector Classifier', SVC(kernel = 'rbf')))
        models.append(('Decision Tree Regressor', DecisionTreeRegressor()))
        models.append(('Random Forest Classifier', RandomForestClassifier(n_estimators = 10)))

        # evaluate each model in turn
        results = []
        names = []
        scoring = 'neg_mean_absolute_error'
        for name, model in models:
            kfold = model_selection.KFold(n_splits=4, random_state=seed)
            cv_results = model_selection.cross_val_score(model, X_train, y_train.ravel(), cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

        # boxplot algorithm comparison
        fig = plt.figure()
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        plt.xticks(rotation=90)
        plt.show()