import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold, KFold
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def file_exists(filepath):
    if os.path.exists(filepath):
        return True
    else:
        return False
    
def rename_filename(filepath, number=1):
    name = os.path.splitext(filepath)[0]
    extension = os.path.splitext(filepath)[1]

    name += "_" + str(number)
    filepath = name + extension
    if file_exists(filepath):
        return rename_filename(filepath=filepath, number=number+1)
    else:
        return filepath

def save_model(to_save, filepath):
    import pickle

    try:
        if file_exists(filepath=filepath):
            filepath = rename_filename(filepath=filepath)
        pickle.dump(to_save, open(filepath, 'wb'))
        print("Saved successfully")
        return True, filepath
    except Exception as e:
        print("Error during saving model:\n", e)
        return False, filepath

def choose_model(option_user, **params):
    if int(option_user) == 1:
        model_option = input("Which model do you want to use?                                                                              1 = LinearRegression 2 = PolynomialFeatures 3 = SVM - SVR                                                    4 =  RandomForestRegressor")
        option_user = int(model_option)
        if option_user == 1:
            from sklearn.linear_model import LinearRegression
            if params:
                for k,v in params.items():
                    model = LinearRegression(**v) 
                return model
            else:
                model = LinearRegression()
                return model

        if option_user == 2:
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression
            if params:
                for k,v in params.items():
                    model = PolynomialFeatures(**v) 
                return model
            else:
                raise ValueError("Missing argument degree")
            
        if option_user == 3:
            from sklearn.svm import SVR
            if params:
                for k,v in params.items():
                    model = SVR(**v) 
                return model
            else:
                model = SVR()
                return model

        if option_user == 4:
            from sklearn.ensemble import RandomForestRegressor
            if params:
                for k,v in params.items():
                    model = RandomForestRegressor(**v) 
                return model
            else:
                model = RandomForestRegressor()
                return model

        return model 
        
    elif int(option_user) == 2:
        model_option = input("Which model do you want to use? 1 = LogisticRegression, 2 = svm - SVC, 3 =                                  KNeighborsClassifier 4 = RandomForestClassifier(), 5 = XGBClassifier()")
        
        option_user = int(model_option)

        if option_user == 1:
            from sklearn.linear_model import LogisticRegression
            if params:
                for k,v in params.items():
                    model = LogisticRegression(**v) 
                return model
            else:
                model = LogisticRegression()
                return model


        if option_user == 2:
            from sklearn import svm
            if params:
                for k,v in params.items():
                    model = svm.SVC(**v) 
                return model
            else:
                model = svm.SVC()
                return model

        if option_user == 3:
            from sklearn.neighbors import KNeighborsClassifier
            for k,v in params.items():
                model = KNeighborsClassifier(**v) 
                return model
            else:
                raise ValueError("Missing argument n_neighbors")

        if option_user == 4:
            from sklearn.ensemble import RandomForestClassifier
            if params:
                for k,v in params.items():
                    model = RandomForestClassifier(**v) 
                return model
            else:
                model = RandomForestClassifier()
                return model

        if option_user == 5:
            from xgboost import XGBClassifier
            for k,v in params.items():
                model = XGBClassifier(**v) 
                return model
            else:
                model = XGBClassifier()
                return model

def train_model(model, df, target_name):
    X = df.drop(target_name, 1).values
    y = df[target_name].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=4)
    
    kfold_train = input("Do you want cross validation? yes or no")
    if kfold_train.lower() != "yes":
        """
    Starting training process with all X_train data

        """
    
        if str(model).startswith("PolynomialFeatures"):
            
            X_poly = model.fit_transform(X_train, y_train)

            many_linear = (input('Enter the number of parameters you want to give for LinearRegression, if you dont want any, enter: no. '))
            param_list_linear = {}
            if many_linear.lower == "no":
                lin_reg_model = LinearRegression()
            else:
                for i in range(int(many_linear)):
                    data = input('Enter parameter & value separated by ":" ') 
                    temp = data.split(':') 
                    if temp[1].isdigit():
                        param_list_linear[temp[0]] = int(temp[1]) 
                    elif ("True" in temp[1])|("False" in temp[1]):
                        param_list_linear[temp[0]] = bool(temp[1])
                    else:
                        param_list_linear[temp[0]] = temp[1]
                if param_list_linear:
                    lin_reg_model = LinearRegression(**param_list_linear) 
                    

            model_trained = lin_reg_model.fit(X_poly, y_train)

            X_test_poly = model.fit_transform(X_test, y_test)
            accuracy = model_trained.score(X_test_poly, y_test)

        else:
            model_trained = model.fit(X_train, y_train)
            accuracy = model_trained.score(X_test, y_test)
    else:
        small_portions = input("Do you want to use cross validation in small steps? (for large datasets), put yes or no.")
        if small_portions == "no":
            """
            Starting the training with cross validation normally
            """
            if str(model).startswith("PolynomialFeatures"):
                    n_splits = int(input("Put number of n_splits:"))
                    n_repeats = int(input("Put the number of n_repeats:"))
                    k_fold = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=4)
                    val_score = []
                    train_score = []
                    
                    X_poly = model.fit_transform(X_train, y_train)

                    many_linear_1 = input('Enter the number of parameters you want to give for LinearRegression, if you dont want any, enter: no. ')
                    
                    param_list_linear = {}
                    if many_linear_1 == "no":
                        
                        lin_reg_model = LinearRegression()
                    else:
                        for i in range(int(many_linear_1)):
                            data = input('Enter parameter & value separated by ":" ') 
                            temp = data.split(':') 
                            if temp[1].isdigit():
                                param_list_linear[temp[0]] = int(temp[1]) 
                            elif ("True" in temp[1])|("False" in temp[1]):
                                param_list_linear[temp[0]] = bool(temp[1])
                            else:
                                param_list_linear[temp[0]] = temp[1]
                        if param_list_linear:
                            lin_reg_model = LinearRegression(**param_list_linear) 
                            
                    for i, (train, val) in enumerate(k_fold.split(X_poly)):
                        model_trained = lin_reg_model.fit(X_poly[train], y_train[train])
                        score_val = lin_reg_model.score(X_poly[val], y_train[val])
                        val_score.append(score_val)
                        score_train = lin_reg_model.score(X_poly[train], y_train[train])
                        train_score.append(score_train)
                    #model_trained = lin_reg_model.fit(X_poly, y_train)

                    X_test_poly = model.fit_transform(X_test, y_test)
                    accuracy = model_trained.score(X_test_poly, y_test)
            else:
                n_splits = int(input("Put number of n_splits:"))
                n_repeats = int(input("Put the number of n_repeats:"))
                k_fold = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=4)
                val_score = []
                train_score = []
                for i, (train, val) in enumerate(k_fold.split(X_train)):
                    model_trained = model.fit(X_train[train], y_train[train])
                    score_val = model.score(X_train[val], y_train[val])
                    val_score.append(score_val)
                    score_train = model.score(X_train[train], y_train[train])
                    train_score.append(score_train)

                accuracy = model_trained.score(X_test, y_test)

            
            print("showing the learning process")
            plt.plot(train_score, label="train")
            plt.plot(val_score, label="val", color="orange")
            plt.ylabel("score")
            plt.legend()
            plt.show()
        else:
            """
            Starting cross validation with small steps - for large datasets. 
            """
            if str(model).startswith("PolynomialFeatures"):
                val_score = []
                train_score = []
                scores_small_trains = []
                scores_small_vals = []
                scores_smalls_trains_iterations = []
                scores_smalls_vals_iterations = []

                
                n_splits = int(input("Put number of n_splits:"))
                n_repeats = int(input("Put the number of n_repeats:"))
                k_fold = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=4)
                n_small_splits = int(input("put number of n_splits for the small sections to train in cross validation"))
                kfold_small_trains = KFold(n_splits=n_small_splits, random_state=4)
                """
                Per model the warm_state parameter is different (if they even excist). Look for documentation how to apply                        warm state for cross validation learning. below it will ask if you want to apply partial_fit
                """
                partial_fit = input("apply partial_fit to model?, put yes or no")
    
                
                X_poly = model.fit_transform(X_train, y_train)

                many_linear_1 = input('Enter the number of parameters you want to give for LinearRegression, if you dont want any, enter: no. ')
                param_list_linear = {}
                if many_linear_1 == "no":
                    lin_reg_model = LinearRegression()
                else:
                    for i in range(int(many_linear_1)):
                        data = input('Enter parameter & value separated by ":" ') 
                        temp = data.split(':') 
                        if temp[1].isdigit():
                            param_list_linear[temp[0]] = int(temp[1]) 
                        elif ("True" in temp[1])|("False" in temp[1]):
                            param_list_linear[temp[0]] = bool(temp[1])
                        else:
                            param_list_linear[temp[0]] = temp[1]
                    if param_list_linear:
                        lin_reg_model = LinearRegression(**param_list_linear) 

                for i, (train, val) in enumerate(k_fold.split(X_poly)):
                    to_show_in_bar = ": " + str(i) + "/" + str(n_splits * n_repeats)
                    generator_val = kfold_small_trains.split(val)
                    

                    for i2,(_, small_train) in tqdm(enumerate(kfold_small_trains.split(train)), total=n_small_splits, desc="Small train  progress" + to_show_in_bar):
                        _, small_val = next(generator_val)
                            

                        if partial_fit.lower() == "yes":
                            model_trained = lin_reg_model.partial_fit(X_poly[small_train], y_train[small_train], classes=np.unique(y))
                        else:
                            model_trained = lin_reg_model.fit(X_poly[small_train], y_train[small_train])

                        score_small_train = model.score(X_poly[small_train], y_train[small_train])
                        scores_smalls_trains_iterations.append(score_small_train)
                        # val part
                        score_small_val = model.score(X_poly[small_val], y_train[small_val])
                        scores_smalls_vals_iterations.append(score_small_val)
                    
                        train_score.append(np.mean(scores_smalls_trains_iterations))
                        scores_small_trains = scores_small_trains + list(scores_smalls_trains_iterations)
                        scores_smalls_trains_iterations.clear()
                        val_score.append(np.mean(scores_smalls_vals_iterations))
                        scores_small_vals = scores_small_vals + list(scores_smalls_vals_iterations)
                        scores_smalls_vals_iterations.clear()

                    print("Iteration:", to_show_in_bar, "| Val_accuracy:", np.mean(val_score),                                          "| train_accuracy: ", np.mean(train_score), sep="~~~~~~")
                print("Training finished!")

            else:
                val_score = []
                train_score = []
                scores_small_trains = []
                scores_small_vals = []
                scores_smalls_trains_iterations = []
                scores_smalls_vals_iterations = []

                
                n_splits = int(input("Put number of n_splits:"))
                n_repeats = int(input("Put the number of n_repeats:"))
                k_fold = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=4)
                n_small_splits = int(input("put number of n_splits for the small sections to train in cross                                      validation"))
                
                kfold_small_trains = KFold(n_splits=n_small_splits, random_state=4)
                """
                Per model the warm_state parameter is different (if they even excist). Look for documentation how to apply                        warm state for cross validation learning. below it will ask if you want to apply partial_fit"""
                partial_fit = input("apply partial_fit to model?, put yes or no")

                for i, (train, val) in enumerate(k_fold.split(X_train)):
                    to_show_in_bar = ": " + str(i) + "/" + str(n_splits * n_repeats)
                    generator_val = kfold_small_trains.split(val)

                    for i2,(_, small_train) in tqdm(enumerate(kfold_small_trains.split(train)), total=n_small_splits, desc="Small train  progress" + to_show_in_bar):
                        _, small_val = next(generator_val)
                                

                        if partial_fit.lower() == "yes":
                            model_trained = model.partial_fit(X_train[small_train], y_train[small_train], classes=np.unique(y))
                        else:
                            model_trained = model.fit(X_train[small_train], y_train[small_train])

                        score_small_train = model.score(X_train[small_train], y_train[small_train])
                        scores_smalls_trains_iterations.append(score_small_train)
                        # val part
                        score_small_val = model.score(X_train[small_val], y_train[small_val])
                        scores_smalls_vals_iterations.append(score_small_val)
                        train_score.append(np.mean(scores_smalls_trains_iterations))
                        scores_small_trains = scores_small_trains + list(scores_smalls_trains_iterations)
                        scores_smalls_trains_iterations.clear()
                        val_score.append(np.mean(scores_smalls_vals_iterations))
                        scores_small_vals = scores_small_vals + list(scores_smalls_vals_iterations)
                        scores_smalls_vals_iterations.clear()

                        print("Iteration:", to_show_in_bar, "| Val_accuracy:", np.mean(val_score), "| train_accuracy: ",                                        np.mean(train_score), sep="~~~~~~")
                print("Training finished!")
                    
            accuracy = model_trained.score(X_test, y_test)
            
            print("showing the learning process")
            plt.plot(train_score, label="train")
            plt.plot(val_score, label="val", color="orange")
            plt.ylabel("score")
            plt.legend()
            plt.show()


        
    return model_trained, accuracy

def main(df):
    """
    Function that will create model of choice with parameters of choice and trains it with desired method. 
    After showing the learning process and showing the accuracy score, the model can be saved for further usage.

    Function receives one parameter: the dataframe. The dataframe should contain data that has been cleaned and filtered
    with the desired columns including the target column. 
    """
    '''
    for regression: 
        option 1 = LinearRegression
        option 2 = PolynomialFeatures
        option 3 = SVM - SVR
        option 4 = RandomForestRegressor
    for classification: 
        option 1 = LogisticRegression
        option 2 = KNeighborsClassifier
        option 3 = svm - SVC
        option 4 = RandomForestClassifier()
        option 5 = XGBClassifier()
    '''

    choice = input("What type of problem: 1 for regression or 2 for classification?")
    params = input("Enter YES in case you want to enter a dictionary of params, if not neccesary put NO") 
    target = input("What is the target column?")

    if params.lower() == "no":
        model = choose_model(option_user=choice)
        model_trained, accuracy = train_model(model=model, df=df, target_name=target)
    else:
        many = int(input('Enter the number of parameters you want to give: '))
        param_list = {}
        for i in range(many):
            data = input('Enter parameter & value separated by ":" ') 
            temp = data.split(':') 
            if temp[1].isdigit():
                param_list[temp[0]] = int(temp[1]) 
            elif ("True" in temp[1])|("False" in temp[1]):
                param_list[temp[0]] = bool(temp[1])
            else:
                param_list[temp[0]] = temp[1]

        model = choose_model(option_user=choice, params=param_list)
        model_trained, accuracy = train_model(model=model, df=df, target_name=target)

    import time
    print("score of model:", accuracy)
    

    time.sleep(3.5)    # pause 3.5 seconds

    save = input("Do you want to save the model?, put yes or no")

    if save.lower() == "yes":
        filepath = input("Put the filepath where you want to save the model as following: name_file.sav ")
        save_model(to_save=model_trained, filepath=filepath)
    
    
    return model_trained
    


    



  