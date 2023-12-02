import random
import pandas as pd
import numpy as np
import os 
import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import balanced_accuracy_score, make_scorer, recall_score, roc_curve, auc
from aif360.sklearn.metrics import disparate_impact_ratio, average_odds_difference, statistical_parity_difference, equal_opportunity_difference

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from under_and_over import undersampling_dataset, oversampling_subgroups, oversampling_all_data
from oversampling import oversampling_df
from scipy import interp

RND_SEED = 44
RESULTS_LOCAL = 'results'

def run_cross_validation_tests(X_data, y_data, dict_weight, algorithm_name, df_name, flg_oversampling_groups=True):
    algorithms = {
        "kNN": (KNeighborsClassifier(p=2), {'n_neighbors': [1,5,10]}),
        "SVM" : (SVC(random_state=RND_SEED), {'kernel': ['linear', 'rbf']}),
        "RF" : (RandomForestClassifier(random_state=RND_SEED), {'max_depth': [5, 10, 20]}),
        "LSR": (LogisticRegression(random_state=RND_SEED), {'C': [1.0], 'max_iter': [50, 100, 150, 200]}),
    }

    #10-fold cross validation 
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RND_SEED)

    #define Standard Scaler to standardize the features
    prep = StandardScaler()

    #3 folds to choose the best hyperparameters
    gskf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RND_SEED)

    #choose of the best hyperparameters through balanced accuracy
    perf = balanced_accuracy_score

    #store the recall of each algorithm 
    score = {}
    for algorithm in algorithms.keys():
        score[algorithm] = []

    acc_score = {}
    for algorithm in algorithms.keys():
        acc_score[algorithm] = []
    
    f1_dict_score = {}
    for algorithm in algorithms.keys():
        f1_dict_score[algorithm] = []

    precision_dict_score = {}
    for algorithm in algorithms.keys():
        precision_dict_score[algorithm] = []

    roc_score = {}
    for algorithm in algorithms.keys():
        roc_score[algorithm] = []

    auc_score = {}
    for algorithm in algorithms.keys():
        auc_score[algorithm] = []

    ######################################################
    #store results for protected attributes
    aod_1_dict = {}
    for algorithm in algorithms.keys():
        aod_1_dict[algorithm] = []

    eod_1_dict = {}
    for algorithm in algorithms.keys():
        eod_1_dict[algorithm] = []

    spd_1_dict = {}
    for algorithm in algorithms.keys():
        spd_1_dict[algorithm] = []

    di_1_dict = {}
    for algorithm in algorithms.keys():
        di_1_dict[algorithm] = []

    # protected attribute 2 if exists
    aod_2_dict = {}
    for algorithm in algorithms.keys():
        aod_2_dict[algorithm] = []

    eod_2_dict = {}
    for algorithm in algorithms.keys():
        eod_2_dict[algorithm] = []

    spd_2_dict = {}
    for algorithm in algorithms.keys():
        spd_2_dict[algorithm] = []

    di_2_dict = {}
    for algorithm in algorithms.keys():
        di_2_dict[algorithm] = []

    ######################################################
    #store results for prottected algorithm
    ####### Blacks #######
    recall_score_blacks = {}
    for algorithm in algorithms.keys():
        recall_score_blacks[algorithm] = []

    acc_score_blacks = {}
    for algorithm in algorithms.keys():
        acc_score_blacks[algorithm] = []

    f1_dict_score_blacks = {}
    for algorithm in algorithms.keys():
        f1_dict_score_blacks[algorithm] = []

    precision_dict_score_blacks = {}
    for algorithm in algorithms.keys():
        precision_dict_score_blacks[algorithm] = []

    ####### Whites #######
    recall_score_whites = {}
    for algorithm in algorithms.keys():
        recall_score_whites[algorithm] = []

    acc_score_whites = {}
    for algorithm in algorithms.keys():
        acc_score_whites[algorithm] = []

    f1_dict_score_whites = {}
    for algorithm in algorithms.keys():
        f1_dict_score_whites[algorithm] = []

    precision_dict_score_whites = {}
    for algorithm in algorithms.keys():
        precision_dict_score_whites[algorithm] = []


    ####### Male #######
    recall_score_male = {}
    for algorithm in algorithms.keys():
        recall_score_male[algorithm] = []

    acc_score_male = {}
    for algorithm in algorithms.keys():
        acc_score_male[algorithm] = []

    f1_dict_score_male = {}
    for algorithm in algorithms.keys():
        f1_dict_score_male[algorithm] = []

    precision_dict_score_male = {}
    for algorithm in algorithms.keys():
        precision_dict_score_male[algorithm] = []

    ####### Female #######
    recall_score_female = {}
    for algorithm in algorithms.keys():
        recall_score_female[algorithm] = []

    acc_score_female = {}
    for algorithm in algorithms.keys():
        acc_score_female[algorithm] = []

    f1_dict_score_female = {}
    for algorithm in algorithms.keys():
        f1_dict_score_female[algorithm] = []

    precision_dict_score_female = {}
    for algorithm in algorithms.keys():
        precision_dict_score_female[algorithm] = []


    ######################################################

    #for each algorithm and its respective search space
    for algorithm, (clf, parameters) in algorithms.items():
        print('Dataset: ', df_name)
        print(f'Algorithm: {algorithm_name} - {algorithm}', )
        
        #define a grid search for the best hyperparameters
        best = GridSearchCV(clf, parameters, cv=gskf, scoring=make_scorer(perf))
        for train, test in kf.split(X_data, y_data):
            
            #split train and test 
            X_train, X_test = X_data.iloc[train], X_data.iloc[test]
            y_train, y_test = y_data.iloc[train], y_data.iloc[test]

            # Oversampling
            df_train = X_train.join(y_train)

            if algorithm_name == 'FairADASYN':
                # Undersampling (Não Favorável e Privilegiado)
                df_under = undersampling_dataset(df_train, df_name)
                df_train = df_under.copy()

            # Oversampling subgroups or all data
            if flg_oversampling_groups:
                df_oversampled = oversampling_df(df_train, dict_weight, algorithm_name, df_name)
                #df_oversampled = oversampling_subgroups(df_train, dict_weight, algorithm_name, df_name)
            else:
                df_oversampled = oversampling_all_data(df_train, dict_weight, algorithm_name, df_name)

            X_train = df_oversampled.drop(['Probability'], axis=1)
            y_train = df_oversampled['Probability']

            #vectors to store y_pred e y_true
            y_pred = [] 
            y_true = [] 

            #standardize the features                
            prep.fit(X_train)
            #search for the best hyperparameters
            best.fit(prep.transform(X_train), y_train)

            print(best.best_params_)
            #store the results
            y_pred = [*y_pred, *(best.predict(prep.transform(X_test)))] 
            #print("[y_true, y_test]:",[y_true, y_test])

            y_true =  [*y_true, *y_test]
            #print("[*y_true, *y_test]:",y_true)

            #calculate the recall
            #print('y_true: ',y_true)
            print(len(y_true),len(y_pred))
            score[algorithm].append(recall_score(y_true, y_pred, average = None))
            #calculate accuracy
            acc_score[algorithm].append( accuracy_score(y_true, y_pred) )
            #calculate f1
            f1_dict_score[algorithm].append( f1_score(y_true, y_pred) )
            #calculate precision
            precision_dict_score[algorithm].append( precision_score(y_true, y_pred) )
            #calculate roc score
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
            roc_score[algorithm].append( interp(np.linspace(0,1,100), fpr, tpr) )
            auc_score[algorithm].append( roc_auc )

            ############ ====== Fairness Metrics ===== ##############
            protected_attributes = list(dict_weight.keys())
            df_test = X_test.join(y_test)

            ############# Ajusta "y_teste" para utilizar lib #############

            df_test_mult_indexes = df_test.copy()
            df_test_mult_indexes = df_test_mult_indexes.set_index(protected_attributes)

            y_test_mult_index = df_test_mult_indexes['Probability']
            y_pred = np.array(y_pred)
  
            # Protected Attribute 1
            aod_1_dict[algorithm].append( round(average_odds_difference(y_test_mult_index, y_pred, prot_attr=protected_attributes[0]), 2) )

            eod_1_dict[algorithm].append( round(equal_opportunity_difference(y_test_mult_index, y_pred, prot_attr=protected_attributes[0]),2 ) )

            spd_1_dict[algorithm].append( round(statistical_parity_difference(y_test_mult_index, y_pred, prot_attr=protected_attributes[0]), 2) )

            di_1_dict[algorithm].append( round(disparate_impact_ratio(y_test_mult_index, y_pred, prot_attr=protected_attributes[0]), 2) )

            #############################################################

            if df_name == 'Adult' or df_name == 'Compas':
                # Protected Attribute 2
                aod_2_dict[algorithm].append( round(average_odds_difference(y_test_mult_index, y_pred, prot_attr=protected_attributes[1]), 2) )

                eod_2_dict[algorithm].append( round(equal_opportunity_difference(y_test_mult_index, y_pred, prot_attr=protected_attributes[1]),2 ) )

                spd_2_dict[algorithm].append( round(statistical_parity_difference(y_test_mult_index, y_pred, prot_attr=protected_attributes[1]), 2) )

                di_2_dict[algorithm].append( round(disparate_impact_ratio(y_test_mult_index, y_pred, prot_attr=protected_attributes[1]), 2) )

            ##########################################################

            # Results per protected attribute
            if df_name == 'Adult' or df_name == 'Compas':
                data_afro = X_test[X_test['race'] == 0]
                data_white = X_test[X_test['race'] == 1]

                data_male = X_test[X_test['sex'] == 1]
                data_female = X_test[X_test['sex'] == 0]

                # Separe into two races
                y_test_afro = y_test.loc[data_afro.index]
                y_test_white = y_test.loc[data_white.index]

                y_pred_black = [] 
                y_true_black = [] 
                
                y_pred_black = [*y_pred_black, *(best.predict(prep.transform(data_afro)))] 
                y_true_black =  [*y_true_black, *y_test_afro]

                # Metrics for Black
                acc_score_blacks[algorithm].append(accuracy_score(y_true=y_true_black, y_pred = y_pred_black))
                precision_dict_score_blacks[algorithm].append(precision_score(y_true=y_true_black, y_pred = y_pred_black))
                recall_score_blacks[algorithm].append(recall_score(y_true=y_true_black, y_pred = y_pred_black, average = None))
                f1_dict_score_blacks[algorithm].append(f1_score(y_true=y_true_black, y_pred = y_pred_black))


                # Data White
                y_pred_white = [] 
                y_true_white = []

                y_pred_white = [*y_pred_white, *(best.predict(prep.transform(data_white)))] 
                y_true_white =  [*y_true_white, *y_test_white]

                # Metrics for Whites
                acc_score_whites[algorithm].append(accuracy_score(y_true=y_true_white, y_pred = y_pred_white))
                precision_dict_score_whites[algorithm].append(precision_score(y_true=y_true_white, y_pred = y_pred_white))
                recall_score_whites[algorithm].append(recall_score(y_true=y_true_white, y_pred = y_pred_white, average = None))
                f1_dict_score_whites[algorithm].append(f1_score(y_true=y_true_white, y_pred = y_pred_white))

                # Separe into sex
                y_test_male = y_test.loc[data_male.index]
                y_test_female = y_test.loc[data_female.index]

                y_pred_male = [] 
                y_true_male = []

                y_pred_male = [*y_pred_male, *(best.predict(prep.transform(data_male)))] 
                y_true_male =  [*y_true_male, *y_test_male]

                # Metrics for Male
                acc_score_male[algorithm].append(accuracy_score(y_true=y_true_male, y_pred = y_pred_male))
                precision_dict_score_male[algorithm].append(precision_score(y_true=y_true_male, y_pred = y_pred_male))
                recall_score_male[algorithm].append(recall_score(y_true=y_true_male, y_pred = y_pred_male, average = None))
                f1_dict_score_male[algorithm].append(f1_score(y_true=y_true_male, y_pred = y_pred_male))


                # Data Female
                y_pred_female = [] 
                y_true_female = []

                y_pred_female = [*y_pred_female, *(best.predict(prep.transform(data_female)))] 
                y_true_female =  [*y_true_female, *y_test_female]

                # Metrics for Female
                acc_score_female[algorithm].append(accuracy_score(y_true=y_true_female, y_pred = y_pred_female))
                precision_dict_score_female[algorithm].append(precision_score(y_true=y_true_female, y_pred = y_pred_female))
                recall_score_female[algorithm].append(recall_score(y_true=y_true_female, y_pred = y_pred_female, average = None))
                f1_dict_score_female[algorithm].append(f1_score(y_true=y_true_female, y_pred = y_pred_female))

            ##########################################################
            else:
                data_male = X_test[X_test[protected_attributes[0]] == 1]
                data_female = X_test[X_test[protected_attributes[0]] == 0]

                # Separe into sex
                y_test_male = y_test.loc[data_male.index]
                y_test_female = y_test.loc[data_female.index]

                y_pred_male = [] 
                y_true_male = []

                y_pred_male = [*y_pred_male, *(best.predict(prep.transform(data_male)))] 
                y_true_male =  [*y_true_male, *y_test_male]

                # Metrics for Male
                acc_score_male[algorithm].append(accuracy_score(y_true=y_true_male, y_pred = y_pred_male))
                precision_dict_score_male[algorithm].append(precision_score(y_true=y_true_male, y_pred = y_pred_male))
                recall_score_male[algorithm].append(recall_score(y_true=y_true_male, y_pred = y_pred_male, average = None))
                f1_dict_score_male[algorithm].append(f1_score(y_true=y_true_male, y_pred = y_pred_male))

                # Data Female
                y_pred_female = [] 
                y_true_female = []

                y_pred_female = [*y_pred_female, *(best.predict(prep.transform(data_female)))] 
                y_true_female =  [*y_true_female, *y_test_female]

                # Metrics for Female
                acc_score_female[algorithm].append(accuracy_score(y_true=y_true_female, y_pred = y_pred_female))
                precision_dict_score_female[algorithm].append(precision_score(y_true=y_true_female, y_pred = y_pred_female))
                recall_score_female[algorithm].append(recall_score(y_true=y_true_female, y_pred = y_pred_female, average = None))
                f1_dict_score_female[algorithm].append(f1_score(y_true=y_true_female, y_pred = y_pred_female))
            
        print('________________________')
        print(best.best_estimator_)
        print('________________________')

    #write a csv with the recall of class '0' - specificity 
    #and another csv with the recall of class '1' - sensitivity
    
    recall_knn = pd.DataFrame(np.vstack(score['kNN']))
    recall_adpd = pd.DataFrame(np.vstack(score['SVM']))
    recall_adpnd = pd.DataFrame(np.vstack(score['RF']))
    recall_lsr = pd.DataFrame(np.vstack(score['LSR']))

    esp = pd.concat([ recall_knn[[0]], recall_adpd[[0]], recall_adpnd[[0]], recall_lsr[[0]] ], axis=1)
    sen = pd.concat([ recall_knn[[1]], recall_adpd[[1]], recall_adpnd[[1]], recall_lsr[[1]] ], axis=1)

    esp.columns = ['kNN', 'SVM', 'RF', 'LSR']
    sen.columns = ['kNN', 'SVM', 'RF', 'LSR']

    # Create Dataset folder to store results
    if os.path.exists(f'results/{df_name} Dataset'):
        pass
    else:
        os.mkdir(f'results/{df_name} Dataset')

    # Create Algorithm folder to store results
    if os.path.exists(f'results/{df_name} Dataset/{algorithm_name}'):
        pass
    else:
        os.mkdir(f'results/{df_name} Dataset/{algorithm_name}')

    esp.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_spe.csv')
    sen.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_sen.csv')  

    #write df with accuracy score
    df_acc_score = pd.DataFrame.from_dict(acc_score)
    df_acc_score.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_accuracy.csv')
    
    #write df with f1 score
    df_f1_score = pd.DataFrame.from_dict(f1_dict_score)
    df_f1_score.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_f1.csv')

    #write df with precision score
    df_precision_score = pd.DataFrame.from_dict(precision_dict_score)
    df_precision_score.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_precision.csv')

    #write df with roc score
    df_roc_score = pd.DataFrame.from_dict(roc_score)
    df_roc_score.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_roc_score.csv')

    #write df with auc score
    df_auc_score = pd.DataFrame.from_dict(auc_score)
    df_auc_score.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_auc.csv')

    #################### FAIRNESS METRICS PER PROTECTED ATTRIBUTE ####################
    df_aod = pd.DataFrame.from_dict(aod_1_dict)
    df_aod.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + f'_aod_{protected_attributes[0]}.csv')

    df_eod = pd.DataFrame.from_dict(eod_1_dict)
    df_eod.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + f'_eod_{protected_attributes[0]}.csv')

    df_spd = pd.DataFrame.from_dict(spd_1_dict)
    df_spd.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + f'_spd_{protected_attributes[0]}.csv')

    df_di = pd.DataFrame.from_dict(di_1_dict)
    df_di.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + f'_di_{protected_attributes[0]}.csv')

    if df_name == 'Adult' or df_name == 'Compas':
        df_aod = pd.DataFrame.from_dict(aod_2_dict)
        df_aod.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + f'_aod_{protected_attributes[1]}.csv')

        df_eod = pd.DataFrame.from_dict(eod_2_dict)
        df_eod.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + f'_eod_{protected_attributes[1]}.csv')

        df_spd = pd.DataFrame.from_dict(spd_2_dict)
        df_spd.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + f'_spd_{protected_attributes[1]}.csv')

        df_di = pd.DataFrame.from_dict(di_2_dict)
        df_di.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + f'_di_{protected_attributes[1]}.csv')

    #################### RESULTS PER PROTECTED ATTRIBUTE ####################

    if df_name == 'Adult' or df_name == 'Compas':
        recall_blacks_knn = pd.DataFrame(np.vstack(recall_score_blacks['kNN']))
        recall_blacks_adpd = pd.DataFrame(np.vstack(recall_score_blacks['SVM']))
        recall_blacks_adpnd = pd.DataFrame(np.vstack(recall_score_blacks['RF']))
        recall_blacks_lsr = pd.DataFrame(np.vstack(recall_score_blacks['LSR']))

        esp_blacks = pd.concat([ recall_blacks_knn[[0]], recall_blacks_adpd[[0]], recall_blacks_adpnd[[0]], recall_blacks_lsr[[0]] ], axis=1)
        sen_blacks = pd.concat([ recall_blacks_knn[[1]], recall_blacks_adpd[[1]], recall_blacks_adpnd[[1]], recall_blacks_lsr[[1]] ], axis=1)

        esp_blacks.columns = ['kNN', 'SVM', 'RF', 'LSR']
        sen_blacks.columns = ['kNN', 'SVM', 'RF', 'LSR']

        esp_blacks.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_spe_black.csv')
        sen_blacks.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_sen_black.csv')   

        acc_blacks = pd.DataFrame.from_dict(acc_score_blacks)
        acc_blacks.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_accuracy_black.csv')  

        precision_blacks = pd.DataFrame.from_dict(precision_dict_score_blacks)
        precision_blacks.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_precision_black.csv') 

        f1_score_blacks = pd.DataFrame.from_dict(f1_dict_score_blacks)
        f1_score_blacks.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_f1_black.csv') 


        recall_whites_knn = pd.DataFrame(np.vstack(recall_score_whites['kNN']))
        recall_whites_adpd = pd.DataFrame(np.vstack(recall_score_whites['SVM']))
        recall_whites_adpnd = pd.DataFrame(np.vstack(recall_score_whites['RF']))
        recall_whites_lsr = pd.DataFrame(np.vstack(recall_score_whites['LSR']))

        esp_whites = pd.concat([ recall_whites_knn[[0]], recall_whites_adpd[[0]], recall_whites_adpnd[[0]], recall_whites_lsr[[0]] ], axis=1)
        sen_whites = pd.concat([ recall_whites_knn[[1]], recall_whites_adpd[[1]], recall_whites_adpnd[[1]], recall_whites_lsr[[1]] ], axis=1)

        esp_whites.columns = ['kNN', 'SVM', 'RF', 'LSR']
        sen_whites.columns = ['kNN', 'SVM', 'RF', 'LSR']

        esp_whites.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_spe_white.csv')
        sen_whites.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_sen_white.csv')   

        acc_whites = pd.DataFrame.from_dict(acc_score_whites)
        acc_whites.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_accuracy_white.csv')  

        precision_whites = pd.DataFrame.from_dict(precision_dict_score_whites)
        precision_whites.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_precision_white.csv') 

        f1_score_whites = pd.DataFrame.from_dict(f1_dict_score_whites)
        f1_score_whites.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_f1_white.csv') 

        # Metrics for sex
        # Male
        recall_male_knn = pd.DataFrame(np.vstack(recall_score_male['kNN']))
        recall_male_adpd = pd.DataFrame(np.vstack(recall_score_male['SVM']))
        recall_male_adpnd = pd.DataFrame(np.vstack(recall_score_male['RF']))
        recall_male_lsr = pd.DataFrame(np.vstack(recall_score_male['LSR']))

        esp_male = pd.concat([ recall_male_knn[[0]], recall_male_adpd[[0]], recall_male_adpnd[[0]], recall_male_lsr[[0]] ], axis=1)
        sen_male = pd.concat([ recall_male_knn[[1]], recall_male_adpd[[1]], recall_male_adpnd[[1]], recall_male_lsr[[1]] ], axis=1)

        esp_male.columns = ['kNN', 'SVM', 'RF', 'LSR']
        sen_male.columns = ['kNN', 'SVM', 'RF', 'LSR']

        esp_male.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_spe_male.csv')
        sen_male.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_sen_male.csv')   

        acc_male = pd.DataFrame.from_dict(acc_score_male)
        acc_male.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_accuracy_male.csv')  

        precision_male = pd.DataFrame.from_dict(precision_dict_score_male)
        precision_male.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_precision_male.csv') 

        f1_score_male = pd.DataFrame.from_dict(f1_dict_score_male)
        f1_score_male.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_f1_male.csv') 

        # Female
        recall_female_knn = pd.DataFrame(np.vstack(recall_score_female['kNN']))
        recall_female_adpd = pd.DataFrame(np.vstack(recall_score_female['SVM']))
        recall_female_adpnd = pd.DataFrame(np.vstack(recall_score_female['RF']))
        recall_female_lsr = pd.DataFrame(np.vstack(recall_score_female['LSR']))

        esp_female = pd.concat([ recall_female_knn[[0]], recall_female_adpd[[0]], recall_female_adpnd[[0]], recall_female_lsr[[0]] ], axis=1)
        sen_female = pd.concat([ recall_female_knn[[1]], recall_female_adpd[[1]], recall_female_adpnd[[1]], recall_female_lsr[[1]] ], axis=1)

        esp_female.columns = ['kNN', 'SVM', 'RF', 'LSR']
        sen_female.columns = ['kNN', 'SVM', 'RF', 'LSR']

        esp_female.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_spe_female.csv')
        sen_female.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_sen_female.csv')   

        acc_female = pd.DataFrame.from_dict(acc_score_female)
        acc_female.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_accuracy_female.csv')  

        precision_female = pd.DataFrame.from_dict(precision_dict_score_female)
        precision_female.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_precision_female.csv') 

        f1_score_female = pd.DataFrame.from_dict(f1_dict_score_female)
        f1_score_female.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_f1_female.csv')
    
    ########## RESULTS PARA DATAFRAMES COM APENAS 1 ATRIBUTO PROTEGIDO ##########
    # Os resultados indicam sempre em relação ao atributo 'sex' mas não necessariamente será este, era o atributo sensível com mais ocorrência dos datasets testados
    else:
        # Metrics for sex
        # Male
        recall_male_knn = pd.DataFrame(np.vstack(recall_score_male['kNN']))
        recall_male_adpd = pd.DataFrame(np.vstack(recall_score_male['SVM']))
        recall_male_adpnd = pd.DataFrame(np.vstack(recall_score_male['RF']))
        recall_male_lsr = pd.DataFrame(np.vstack(recall_score_male['LSR']))

        esp_male = pd.concat([ recall_male_knn[[0]], recall_male_adpd[[0]], recall_male_adpnd[[0]], recall_male_lsr[[0]] ], axis=1)
        sen_male = pd.concat([ recall_male_knn[[1]], recall_male_adpd[[1]], recall_male_adpnd[[1]], recall_male_lsr[[1]] ], axis=1)

        esp_male.columns = ['kNN', 'SVM', 'RF', 'LSR']
        sen_male.columns = ['kNN', 'SVM', 'RF', 'LSR']

        esp_male.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_spe_male.csv')
        sen_male.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_sen_male.csv')   

        acc_male = pd.DataFrame.from_dict(acc_score_male)
        acc_male.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_accuracy_male.csv')  

        precision_male = pd.DataFrame.from_dict(precision_dict_score_male)
        precision_male.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_precision_male.csv') 

        f1_score_male = pd.DataFrame.from_dict(f1_dict_score_male)
        f1_score_male.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_f1_male.csv') 

        # Female
        recall_female_knn = pd.DataFrame(np.vstack(recall_score_female['kNN']))
        recall_female_adpd = pd.DataFrame(np.vstack(recall_score_female['SVM']))
        recall_female_adpnd = pd.DataFrame(np.vstack(recall_score_female['RF']))
        recall_female_lsr = pd.DataFrame(np.vstack(recall_score_female['LSR']))

        esp_female = pd.concat([ recall_female_knn[[0]], recall_female_adpd[[0]], recall_female_adpnd[[0]], recall_female_lsr[[0]] ], axis=1)
        sen_female = pd.concat([ recall_female_knn[[1]], recall_female_adpd[[1]], recall_female_adpnd[[1]], recall_female_lsr[[1]] ], axis=1)

        esp_female.columns = ['kNN', 'SVM', 'RF', 'LSR']
        sen_female.columns = ['kNN', 'SVM', 'RF', 'LSR']

        esp_female.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_spe_female.csv')
        sen_female.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_sen_female.csv')   

        acc_female = pd.DataFrame.from_dict(acc_score_female)
        acc_female.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_accuracy_female.csv')  

        precision_female = pd.DataFrame.from_dict(precision_dict_score_female)
        precision_female.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_precision_female.csv') 

        f1_score_female = pd.DataFrame.from_dict(f1_dict_score_female)
        f1_score_female.to_csv(f'results/{df_name} Dataset/{algorithm_name}/{algorithm_name}' + '_f1_female.csv')
