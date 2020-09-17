from sklearn.model_selection import GridSearchCV

9
def grid(X, y, model, verbose=None):

    if verbose:
        verbose = 10

    new_model = model

    #param_grid = [
    #   {'penalty' : ['l1', '12',  'elasticnet', 'none'],
    #    'C' : np.logspace(-4, 4, 20),
    #    'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    #    'max_iter' : [100, 1000, 2500, 5000]}
    #]

    logistic_params = {
        'classifier': [LogisticRegression(verbose=verbose, n_jobs=number_of_processors)],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__C': np.logspace(0, 4, 10),
        'classifier__solver': {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'},
        }

    random_forest_params = {
        'classifier': [RandomForestClassifier(verbose=verbose, n_jobs=number_of_processors, warm_start=False)],
        'classifier__n_estimators': [10, 100, 1000, 10000],
        'classifier__max_features': [6,7]
        }

    svm_params = {
        'classifier': [svm.SVC(verbose=False)],
        'classifier__kernel':(['rbf']), 
        'classifier__C':[-10, -1, 0.5, 1,10],
        'classifier__coef0': [-1., 0.1, 0.5, 1, ],
        'classifier__gamma': (['auto'])
        }

    xgboost_params = {
        'classifier': [XGBClassifier(n_jobs=number_of_processors)],
        'classifier__max_delta_step':[0.1, 0, 0.2, 1.],
        'classifier__max_depth': [6, 8, 10, 12, 14]
        }
    # Por si usamos XGBoost
    #from xgboost import plot_importance
    #plot_importance(model)

    knn_params = {
        'classifier': [KNeighborsClassifier()],
        'classifier__n_neighbors':[7, 8, 9, 10, 11],
    }

    # SGDClassifier: utilizar "partial_fit" en vez de "fit" si queremos aprovechar warm_start
    sgc_params = {
        'classifier': [SGDClassifier(verbose=False, n_jobs=number_of_processors, warm_start=True)],
        'classifier__loss':['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
        'classifier__penalty':['l2', 'l1', 'elasticnet'],
        'classifier__alpha':[0.0001,0.00001, 0.001],
        'classifier__learning_rate':['optimal', 'invscaling', 'adaptive'],
        'classifier__eta0':[0.01],
    }

    # Create space of candidate learning algorithms and their hyperparameters
    search_space = [
        #logistic_params,
        #random_forest_params,
        #svm_params,
        #xgboost_params,
        #knn_params,
        sgc_params,
        ]

    # Le podemos poner cualquier clasificador. Irá cambiando según va probando pero necesita 1.
    # Si solo vamos a usar uno, debemos poner aquí que vamos a usar ese.

    # v2, genérico
    pipe = Pipeline(steps=[('classifier', search_space[0]["classifier"][0])])

        clf = GridSearchCV(new_model, param_grid=param_grid, cv=3, n_jobs=-1)

        best_clf = clf.fit(X, y)
        best_clf.best_estimator_