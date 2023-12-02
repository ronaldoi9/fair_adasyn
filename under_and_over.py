import pandas as pd
import numpy as np 
import random

from collections import Counter
from fairSmote import get_ngbr, generate_samples
from fair_adasyn import fair_adasyn

from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE

RND_SEED = 44

def oversampling_subgroups_fairsmote(df_func, protected_attributes, df_name):

    if len(protected_attributes) == 1:
        zero_zero = len(df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 0)])
        zero_one = len(df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 1)])
        one_zero = len(df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 0)])
        one_one = len(df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 1)])

        maximum = max(zero_zero,zero_one,one_zero,one_one)

        if maximum == zero_zero:
            df_zero_zero = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 0)]

            df_zero_one = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 1)]
            df_one_zero = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 0)]
            df_one_one = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 1)]
        if maximum == zero_one:
            df_zero_one = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 1)]

            df_zero_zero = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 0)]
            df_one_zero = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 0)]
            df_one_one = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 1)]
        if maximum == one_zero:
            df_one_zero = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 0)]

            df_zero_zero = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 0)]
            df_zero_one = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 1)]
            df_one_one = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 1)]
        if maximum == one_one:
            df_one_one = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 1)]
            
            df_zero_zero = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 0)]
            df_zero_one = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 1)]
            df_one_zero = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 0)]

        zero_zero_to_be_incresed = maximum - zero_zero ## where both are 0
        zero_one_to_be_incresed = maximum - zero_one ## where class is 0 attribute is 1
        one_zero_to_be_incresed = maximum - one_zero ## where class is 1 attribute is 0
        one_one_to_be_incresed = maximum - one_one ## where class is 1 attribute is 1

        print(zero_zero_to_be_incresed,zero_one_to_be_incresed,one_zero_to_be_incresed,one_one_to_be_incresed)

        df_zero_zero[protected_attributes[0]] = df_zero_zero[protected_attributes[0]].astype(str)

        df_zero_one[protected_attributes[0]] = df_zero_one[protected_attributes[0]].astype(str)

        df_one_zero[protected_attributes[0]] = df_one_zero[protected_attributes[0]].astype(str)

        df_one_one[protected_attributes[0]] = df_one_one[protected_attributes[0]].astype(str)

        df_zero_zero = generate_samples(zero_zero_to_be_incresed,df_zero_zero, df_name)
        df_zero_one = generate_samples(zero_one_to_be_incresed,df_zero_one, df_name)
        df_one_zero = generate_samples(one_zero_to_be_incresed,df_one_zero, df_name)
        df_one_one = generate_samples(one_one_to_be_incresed,df_one_one, df_name)

        df = pd.concat([df_zero_zero,df_zero_one,df_one_zero,df_one_one])
        return df

    elif len(protected_attributes) == 2:
        # first one is class value and second one is 'sex' and third one is 'race'
        zero_zero_zero = len(df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 0)])
        zero_zero_one = len(df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 1)])
        zero_one_zero = len(df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 0)])
        zero_one_one = len(df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 1)])
        one_zero_zero = len(df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 0)])
        one_zero_one = len(df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 1)])
        one_one_zero = len(df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 0)])
        one_one_one = len(df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 1)])

        maximum = max(zero_zero_zero,zero_zero_one,zero_one_zero,zero_one_one,one_zero_zero,one_zero_one,one_one_zero,one_one_one)

        if maximum == zero_zero_zero:
            print("zero_zero_zero is maximum")
            df_zero_zero_zero = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 0)]

            df_zero_zero_one = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 1)]
            df_zero_one_zero = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 0)]
            df_zero_one_one = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 1)]
            df_one_zero_zero = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 0)]
            df_one_zero_one = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 1)]
            df_one_one_zero = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 0)]
            df_one_one_one = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 1)]
        if maximum == zero_zero_one:
            print("zero_zero_one is maximum")
            df_zero_zero_one = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 1)]

            df_zero_zero_zero = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 0)]
            df_zero_one_zero = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 0)]
            df_zero_one_one = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 1)]
            df_one_zero_zero = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 0)]
            df_one_zero_one = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 1)]
            df_one_one_zero = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 0)]
            df_one_one_one = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 1)]
        if maximum == zero_one_zero:
            print("zero_one_zero is maximum")
            df_zero_one_zero = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 0)]

            df_zero_zero_zero = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 0)]
            df_zero_zero_one = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 1)]
            df_zero_one_one = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 1)]
            df_one_zero_zero = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 0)]
            df_one_zero_one = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 1)]
            df_one_one_zero = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 0)]
            df_one_one_one = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 1)]
        if maximum == zero_one_one:
            print("zero_one_one is maximum")
            df_zero_one_one = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 1)]

            df_zero_zero_zero = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 0)]
            df_zero_zero_one = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 1)]
            df_zero_one_zero = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 0)]
            df_one_zero_zero = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 0)]
            df_one_zero_one = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 1)]
            df_one_one_zero = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 0)]
            df_one_one_one = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 1)]
        if maximum == one_zero_zero:
            print("one_zero_zero is maximum")
            df_one_zero_zero = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 0)]

            df_zero_zero_zero = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 0)]
            df_zero_zero_one = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 1)]
            df_zero_one_zero = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 0)]
            df_zero_one_one = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 1)]
            df_one_zero_one = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 1)]
            df_one_one_zero = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 0)]
            df_one_one_one = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 1)]
        if maximum == one_zero_one:
            print("one_zero_one is maximum")
            df_one_zero_one = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 1)]

            df_zero_zero_zero = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 0)]
            df_zero_zero_one = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 1)]
            df_zero_one_zero = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 0)]
            df_zero_one_one = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 1)]
            df_one_zero_zero = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 0)]
            df_one_one_zero = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 0)]
            df_one_one_one = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 1)]
        if maximum == one_one_zero:
            print("one_one_zero is maximum")
            df_one_one_zero = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 0)]

            df_zero_zero_zero = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 0)]
            df_zero_zero_one = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 1)]
            df_zero_one_zero = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 0)]
            df_zero_one_one = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 1)]
            df_one_zero_zero = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 0)]
            df_one_zero_one = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 1)]
            df_one_one_one = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 1)]
        if maximum == one_one_one:
            print("one_one_one is maximum")
            df_one_one_one = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 1)]

            df_zero_zero_zero = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 0)]
            df_zero_zero_one = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 1)]
            df_zero_one_zero = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 0)]
            df_zero_one_one = df_func[(df_func['Probability'] == 0) & (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 1)]
            df_one_zero_zero = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 0)]
            df_one_zero_one = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 1)]
            df_one_one_zero = df_func[(df_func['Probability'] == 1) & (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 0)]

        zero_zero_zero_to_be_incresed = maximum - zero_zero_zero
        zero_zero_one_to_be_incresed = maximum - zero_zero_one
        zero_one_zero_to_be_incresed = maximum - zero_one_zero
        zero_one_one_to_be_incresed = maximum - zero_one_one
        one_zero_zero_to_be_incresed = maximum - one_zero_zero
        one_zero_one_to_be_incresed = maximum - one_zero_one
        one_one_zero_to_be_incresed = maximum - one_one_zero
        one_one_one_to_be_incresed = maximum - one_one_one

        print(zero_zero_zero_to_be_incresed,zero_zero_one_to_be_incresed,zero_one_zero_to_be_incresed,zero_one_one_to_be_incresed,
        one_zero_zero_to_be_incresed,one_zero_one_to_be_incresed,one_one_zero_to_be_incresed,one_one_one_to_be_incresed)

        df_zero_zero_zero[protected_attributes[0]] = df_zero_zero_zero[protected_attributes[0]].astype(str)
        df_zero_zero_zero[protected_attributes[1]] = df_zero_zero_zero[protected_attributes[1]].astype(str)

        df_zero_zero_one[protected_attributes[0]] = df_zero_zero_one[protected_attributes[0]].astype(str)
        df_zero_zero_one[protected_attributes[1]] = df_zero_zero_one[protected_attributes[1]].astype(str)

        df_zero_one_zero[protected_attributes[0]] = df_zero_one_zero[protected_attributes[0]].astype(str)
        df_zero_one_zero[protected_attributes[1]] = df_zero_one_zero[protected_attributes[1]].astype(str)

        df_zero_one_one[protected_attributes[0]] = df_zero_one_one[protected_attributes[0]].astype(str)
        df_zero_one_one[protected_attributes[1]] = df_zero_one_one[protected_attributes[1]].astype(str)

        df_one_zero_zero[protected_attributes[0]] = df_one_zero_zero[protected_attributes[0]].astype(str)
        df_one_zero_zero[protected_attributes[1]] = df_one_zero_zero[protected_attributes[1]].astype(str)

        df_one_zero_one[protected_attributes[0]] = df_one_zero_one[protected_attributes[0]].astype(str)
        df_one_zero_one[protected_attributes[1]] = df_one_zero_one[protected_attributes[1]].astype(str)

        df_one_one_zero[protected_attributes[0]] = df_one_one_zero[protected_attributes[0]].astype(str)
        df_one_one_zero[protected_attributes[1]] = df_one_one_zero[protected_attributes[1]].astype(str)

        df_one_one_one[protected_attributes[0]] = df_one_one_one[protected_attributes[0]].astype(str)
        df_one_one_one[protected_attributes[1]] = df_one_one_one[protected_attributes[1]].astype(str)

        df_zero_zero_zero = generate_samples(zero_zero_zero_to_be_incresed,df_zero_zero_zero, df_name)
        df_zero_zero_one = generate_samples(zero_zero_one_to_be_incresed,df_zero_zero_one, df_name)
        df_zero_one_zero = generate_samples(zero_one_zero_to_be_incresed,df_zero_one_zero, df_name)
        df_zero_one_one = generate_samples(zero_one_one_to_be_incresed,df_zero_one_one, df_name)
        df_one_zero_zero = generate_samples(one_zero_zero_to_be_incresed,df_one_zero_zero, df_name)
        df_one_zero_one = generate_samples(one_zero_one_to_be_incresed,df_one_zero_one, df_name)
        df_one_one_zero = generate_samples(one_one_zero_to_be_incresed,df_one_one_zero, df_name)
        df_one_one_one = generate_samples(one_one_one_to_be_incresed,df_one_one_one, df_name)

        df = pd.concat([df_zero_zero_zero,df_zero_zero_one,df_zero_one_zero,df_zero_one_one,df_one_zero_zero,df_one_zero_one,df_one_one_zero,df_one_one_one])
        return df

def oversampling_subgroups(df_func, dict_weight, algorithm_name, df_name):  

    label_minority = df_func['Probability'].value_counts().idxmin()
    label_majority = df_func['Probability'].value_counts().idxmax()

    protected_attributes = list(dict_weight.keys())

    if len(protected_attributes) > 1:
        # label = separação 1
        # #'sex'[0] 'race'[1] (Fixa um atributo e varia outro)
        condition_zero_zero = ( (df_func[protected_attributes[0]] == 0) )
        condition_zero_one = ( (df_func[protected_attributes[0]] == 1) )
        condition_one_zero = ( (df_func[protected_attributes[1]] == 0) )
        condition_one_one = ( (df_func[protected_attributes[1]] == 1) )

        # label = separação 2
        # condition_zero_zero = ( (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 0))
        # condition_zero_one =  ( (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 1)) 
        # condition_one_zero =  ( (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 0))
        # condition_one_one =   ( (df_func[protected_attributes[0]] == 1) & (df_func[protected_attributes[1]] == 1))

        df_zero_zero = df_func[condition_zero_zero]
        df_zero_one = df_func[condition_zero_one]
        df_one_zero = df_func[condition_one_zero]
        df_one_one = df_func[condition_one_one]

        print(len(df_zero_zero), len(df_zero_one), len(df_one_zero), len(df_one_one))
        
        if algorithm_name == 'FairADASYN':

            if df_name == 'Adult' or df_name == 'Compas':
                # Desabilita atualização de dict_weight porque tava criando mais amostras de brancos do que pretos
                # Pois com o undersampling, removo muitas pessoas brancas e na hora de criar amostrar pra esse conjunto apenas poucas são geradas
                try:
                    X_res, y_res = fair_adasyn(df_zero_zero.drop(['Probability'],axis=1), df_zero_zero['Probability'], dict_weight, beta=1, K=5, threshold=1, flg_update_dict=False)
                    df_zero_zero = X_res.join(y_res)
                except:
                    df_zero_zero = X_res.join(y_res)

                try:
                    X_res, y_res = fair_adasyn(df_zero_one.drop(['Probability'],axis=1), df_zero_one['Probability'], dict_weight, beta=1, K=5, threshold=1, flg_update_dict=False)
                    df_zero_one = X_res.join(y_res)
                except:
                    df_zero_one = X_res.join(y_res)

                try:
                    X_res, y_res = fair_adasyn(df_one_zero.drop(['Probability'],axis=1), df_one_zero['Probability'], dict_weight, beta=1, K=5, threshold=1, flg_update_dict=False)
                    df_one_zero = X_res.join(y_res)
                except:
                    df_one_zero = X_res.join(y_res)

                try:
                    X_res, y_res = fair_adasyn(df_one_one.drop(['Probability'],axis=1), df_one_one['Probability'], dict_weight, beta=1, K=5, threshold=1, flg_update_dict=False)
                    df_one_one = X_res.join(y_res)
                except:
                    df_one_one = X_res.join(y_res)
            else:
                X_res, y_res = fair_adasyn(df_zero_zero.drop(['Probability'],axis=1), df_zero_zero['Probability'], dict_weight, beta=1, K=5, threshold=1)
                df_zero_zero = X_res.join(y_res)

                X_res, y_res = fair_adasyn(df_zero_one.drop(['Probability'],axis=1), df_zero_one['Probability'], dict_weight, beta=1, K=5, threshold=1)
                df_zero_one = X_res.join(y_res)

                X_res, y_res = fair_adasyn(df_one_zero.drop(['Probability'],axis=1), df_one_zero['Probability'], dict_weight, beta=1, K=5, threshold=1)
                df_one_zero = X_res.join(y_res)

                X_res, y_res = fair_adasyn(df_one_one.drop(['Probability'],axis=1), df_one_one['Probability'], dict_weight, beta=1, K=5, threshold=1)
                df_one_one = X_res.join(y_res)

        elif algorithm_name == 'ADASYN':
            ada = ADASYN(random_state=RND_SEED)

            X_res, y_res = ada.fit_resample(df_zero_zero.drop(['Probability'],axis=1), df_zero_zero['Probability'])
            df_zero_zero = X_res.join(y_res)

            try:
                X_res, y_res = ada.fit_resample(df_zero_one.drop(['Probability'],axis=1), df_zero_one['Probability'])
                df_zero_one = X_res.join(y_res)
            except:
                df_zero_one = X_res.join(y_res)

            try:
                X_res, y_res = ada.fit_resample(df_one_zero.drop(['Probability'],axis=1), df_one_zero['Probability'])
                df_one_zero = X_res.join(y_res)
            except:
                df_one_zero = X_res.join(y_res)

            X_res, y_res = ada.fit_resample(df_one_one.drop(['Probability'],axis=1), df_one_one['Probability'])
            df_one_one = X_res.join(y_res)
            
        elif algorithm_name == 'FairSMOTE':
            df_zero_zero_min = df_zero_zero[df_zero_zero['Probability'] == label_minority]
            df_zero_one_min = df_zero_one[df_zero_one['Probability'] == label_minority]
            df_one_zero_min = df_one_zero[df_one_zero['Probability'] == label_minority]
            df_one_one_min = df_one_one[df_one_one['Probability'] == label_minority]

            # Separa os minoritários e majoritários para fazer o oversampling apenas da classe minoritária
            df_zero_zero_maj = df_zero_zero[df_zero_zero['Probability'] == label_majority]
            df_zero_one_maj = df_zero_one[df_zero_one['Probability'] == label_majority]
            df_one_zero_maj = df_one_zero[df_one_zero['Probability'] == label_majority]
            df_one_one_maj = df_one_one[df_one_one['Probability'] == label_majority]

            zero_zero_to_be_incresed = len(df_zero_zero_maj) - len(df_zero_zero_min)
            zero_one_to_be_incresed = len(df_zero_one_maj) - len(df_zero_one_min)
            one_zero_to_be_incresed = len(df_one_zero_maj) - len(df_one_zero_min)
            one_one_to_be_incresed = len(df_one_one_maj) - len(df_one_one_min)

            print(zero_zero_to_be_incresed,zero_one_to_be_incresed,one_zero_to_be_incresed,one_one_to_be_incresed)

            df_zero_zero_min['race'] = df_zero_zero_min['race'].astype(str)
            df_zero_zero_min['sex'] = df_zero_zero_min['sex'].astype(str)

            df_zero_one_min['race'] = df_zero_one_min['race'].astype(str)
            df_zero_one_min['sex'] = df_zero_one_min['sex'].astype(str)

            df_one_zero_min['race'] = df_one_zero_min['race'].astype(str)
            df_one_zero_min['sex'] = df_one_zero_min['sex'].astype(str)

            df_one_one_min['race'] = df_one_one_min['race'].astype(str)
            df_one_one_min['sex'] = df_one_one_min['sex'].astype(str)

            df_zero_zero_min_over = generate_samples(zero_zero_to_be_incresed,df_zero_zero_min, df_name)
            df_zero_one_min_over = generate_samples(zero_one_to_be_incresed,df_zero_one_min, df_name)
            df_one_zero_min_over = generate_samples(one_zero_to_be_incresed,df_one_zero_min, df_name)
            df_one_one_min_over = generate_samples(one_one_to_be_incresed,df_one_one_min, df_name)

            # Concatena os minoritários que tiveram oversampling com os maj do seu subgrupo
            df_zero_zero = pd.concat([df_zero_zero_min_over, df_zero_zero_maj])
            df_zero_one = pd.concat([df_zero_one_min_over, df_zero_one_maj])
            df_one_zero = pd.concat([df_one_zero_min_over, df_one_zero_maj])
            df_one_one = pd.concat([df_one_one_min_over, df_one_one_maj])

        elif algorithm_name == 'SMOTE':
            sm = SMOTE(random_state=RND_SEED)

            X_res, y_res = sm.fit_resample(df_zero_zero.drop(['Probability'],axis=1), df_zero_zero['Probability'])
            df_zero_zero = X_res.join(y_res)
            
            X_res, y_res = sm.fit_resample(df_zero_one.drop(['Probability'],axis=1), df_zero_one['Probability'])
            df_zero_one = X_res.join(y_res)
        
            X_res, y_res = sm.fit_resample(df_one_zero.drop(['Probability'],axis=1), df_one_zero['Probability'])
            df_one_zero = X_res.join(y_res)
        
            X_res, y_res = sm.fit_resample(df_one_one.drop(['Probability'],axis=1), df_one_one['Probability'])
            df_one_one = X_res.join(y_res)
                

        print(len(df_zero_zero), len(df_zero_one), len(df_one_zero), len(df_one_one))
        df = pd.concat([df_zero_zero,df_zero_one,df_one_zero, df_one_one])

        if df_name == 'Compas':
            cols_to_int = ['sex', 'age_cat', 'race', 'priors_count', 'c_charge_degree', 'Probability']
            df[cols_to_int] = df[cols_to_int].astype(int)

        return df
    else:
        # label = separação 1
        # #'sex'[0] 'race'[1] (Fixa um atributo e varia outro)
        condition_zero_zero = ( (df_func[protected_attributes[0]] == 0) )
        condition_zero_one = ( (df_func[protected_attributes[0]] == 1) )

        
        # label = separação 2
        # condition_zero_zero = ( (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 0))
        # condition_zero_one =  ( (df_func[protected_attributes[0]] == 0) & (df_func[protected_attributes[1]] == 1)) 

        df_zero_zero = df_func[condition_zero_zero]
        df_zero_one = df_func[condition_zero_one]

        print(len(df_zero_zero), len(df_zero_one))

        #maximum = max(len(df_one_one), len(df_zero_one), len(df_one_zero), len(df_one_one))
        
        if algorithm_name == 'FairADASYN':
            try:
                X_res, y_res = fair_adasyn(df_zero_zero.drop(['Probability'],axis=1), df_zero_zero['Probability'], dict_weight, beta=1, K=5, threshold=1)
                df_zero_zero = X_res.join(y_res)
            except:
                print('except -> df_zero_zero')

            try:
                X_res, y_res = fair_adasyn(df_zero_one.drop(['Probability'],axis=1), df_zero_one['Probability'], dict_weight, beta=1, K=5, threshold=1)
                df_zero_one = X_res.join(y_res)
            except:
                print('except -> df_zero_one')

        elif algorithm_name == 'ADASYN':
            ada = ADASYN(random_state=RND_SEED)

            try:
                X_res, y_res = ada.fit_resample(df_zero_zero.drop(['Probability'],axis=1), df_zero_zero['Probability'])
                df_zero_zero = X_res.join(y_res)
            except:
                df_zero_zero = X_res.join(y_res)

            try:
                X_res, y_res = ada.fit_resample(df_zero_one.drop(['Probability'],axis=1), df_zero_one['Probability'])
                df_zero_one = X_res.join(y_res)
            except:
                df_zero_one = X_res.join(y_res)

            
        elif algorithm_name == 'FairSMOTE':
            df_zero_zero_min = df_zero_zero[df_zero_zero['Probability'] == label_minority]
            df_zero_one_min = df_zero_one[df_zero_one['Probability'] == label_minority]

            # Separa os minoritários e majoritários para fazer o oversampling apenas da classe minoritária
            df_zero_zero_maj = df_zero_zero[df_zero_zero['Probability'] == label_majority]
            df_zero_one_maj = df_zero_one[df_zero_one['Probability'] == label_majority]

            zero_zero_to_be_incresed = len(df_zero_zero_maj) - len(df_zero_zero_min)
            zero_one_to_be_incresed = len(df_zero_one_maj) - len(df_zero_one_min)

            print(zero_zero_to_be_incresed,zero_one_to_be_incresed)

            df_zero_zero_min[protected_attributes[0]] = df_zero_zero_min[protected_attributes[0]].astype(str)
            df_zero_one_min[protected_attributes[0]] = df_zero_one_min[protected_attributes[0]].astype(str)

            df_zero_zero_min_over = generate_samples(zero_zero_to_be_incresed,df_zero_zero_min, df_name)
            df_zero_one_min_over = generate_samples(zero_one_to_be_incresed,df_zero_one_min, df_name)

            # Concatena os minoritários que tiveram oversampling com os maj do seu subgrupo
            df_zero_zero = pd.concat([df_zero_zero_min_over, df_zero_zero_maj])
            df_zero_one = pd.concat([df_zero_one_min_over, df_zero_one_maj])

        elif algorithm_name == 'SMOTE':
            sm = SMOTE(random_state=RND_SEED)

            X_res, y_res = sm.fit_resample(df_zero_zero.drop(['Probability'],axis=1), df_zero_zero['Probability'])
            df_zero_zero = X_res.join(y_res)
            
            X_res, y_res = sm.fit_resample(df_zero_one.drop(['Probability'],axis=1), df_zero_one['Probability'])
            df_zero_one = X_res.join(y_res)                

        print(len(df_zero_zero), len(df_zero_one))
        df = pd.concat([df_zero_zero,df_zero_one])

        if df_name == 'German':
            cols_to_int = ['sex', 'age', 'Probability']
            df[cols_to_int] = df[cols_to_int].astype(int)

        return df

def oversampling_all_data(df_func, protected_attributes, algorithm_name, df_name):
    
    if algorithm_name == 'FairADASYN':
        X_res, y_res = fair_adasyn(df_func.drop(['Probability'],axis=1), df_func['Probability'], protected_attributes, beta=1, K=5, threshold=1)

    elif algorithm_name == 'ADASYN':
        ada = ADASYN(random_state=RND_SEED)
        X_res, y_res = ada.fit_resample(df_func.drop(['Probability'],axis=1), df_func['Probability'])
        
    elif algorithm_name == 'FairSMOTE':
        count_unbalance = Counter(df_func["Probability"])

        samples_to_create = abs(count_unbalance[0] - count_unbalance[1])
        df_func['race'] = df_func['race'].astype(str)
        df_func['sex'] = df_func['sex'].astype(str)
        df_res = generate_samples(samples_to_create, df_func, 'Adult')

        cols_float_to_int =  ['age', 'education-num', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'Probability']
        df_res[cols_float_to_int] = df_res[cols_float_to_int].astype(int)
        X_res = df_res.drop(['Probability'],axis=1)
        y_res = df_res['Probability']

    elif algorithm_name == 'SMOTE':
        sm = SMOTE(random_state=RND_SEED)
        X_res, y_res = sm.fit_resample(df_func.drop(['Probability'],axis=1), df_func['Probability'])

    return X_res.join(y_res)
    
def undersampling_dataset(df_func, df_name):

    #Based on class
    df_one , df_zero = [x for _, x in df_func.groupby(df_func['Probability'] == 0)]

    #Based on protected_attribute
    if df_name == 'Adult':
        df_one_white, df_one_nonwhite = [x for _, x in df_one.groupby(df_one['race'] == 0)]
        df_zero_white, df_zero_nonwhite = [x for _, x in df_zero.groupby(df_zero['race'] == 0)]

        nao_fav_priv = (df_zero_white.shape[0]/(df_zero_white.shape[0] + df_zero_nonwhite.shape[0]))

        remove_n = int(len(df_zero_white)*0.05)
        stop_condition = 0.65
        while nao_fav_priv >= stop_condition:
            drop_indices = np.random.choice(df_zero_white.index, remove_n, replace=False)
            df_zero_white.drop(drop_indices, inplace=True)

            nao_fav_priv = (df_zero_white.shape[0]/(df_zero_white.shape[0] + df_zero_nonwhite.shape[0]))

        return pd.concat([df_one_white, df_one_nonwhite, df_zero_white, df_zero_nonwhite])
    elif df_name == 'Compas':
        df_one_white, df_one_nonwhite = [x for _, x in df_one.groupby(df_one['race'] == 0)]
        df_zero_white, df_zero_nonwhite = [x for _, x in df_zero.groupby(df_zero['race'] == 0)]

        nao_fav_nao_priv = (df_zero_nonwhite.shape[0]/(df_zero_white.shape[0] + df_zero_nonwhite.shape[0]))

        remove_n = int(len(df_zero_nonwhite)*0.05)
        stop_condition = 0.65
        while nao_fav_nao_priv >= stop_condition:
            drop_indices = np.random.choice(df_zero_nonwhite.index, remove_n, replace=False)
            df_zero_nonwhite.drop(drop_indices, inplace=True)

            nao_fav_nao_priv = (df_zero_nonwhite.shape[0]/(df_zero_white.shape[0] + df_zero_nonwhite.shape[0]))
        
        return pd.concat([df_one_white, df_one_nonwhite, df_zero_white, df_zero_nonwhite])
    elif df_name == 'German':
        df_one_male, df_one_female = [x for _, x in df_one.groupby(df_one['sex'] == 0)]
        df_zero_male, df_zero_female = [x for _, x in df_zero.groupby(df_zero['sex'] == 0)]

        fav_priv = (df_one_male.shape[0]/df_one.shape[0])

        remove_n = int(len(df_one_male)*0.05)
        stop_condition = 0.60
        while fav_priv >= stop_condition:
            drop_indices = np.random.choice(df_one_male.index, remove_n, replace=False)
            df_one_male.drop(drop_indices, inplace=True)

            fav_priv = (df_one_male.shape[0]/(df_one_male.shape[0] + df_one_female.shape[0]))

        return pd.concat([df_one_male, df_one_female, df_zero_male, df_zero_female])
    elif df_name == 'Default':
        df_one_male, df_one_female = [x for _, x in df_one.groupby(df_one['sex'] == 0)]
        df_zero_male, df_zero_female = [x for _, x in df_zero.groupby(df_zero['sex'] == 0)]

        nao_fav_nao_priv = (df_zero_female.shape[0]/df_zero.shape[0])

        remove_n = int(len(df_zero_female)*0.05)
        stop_condition = 0.50
        while nao_fav_nao_priv >= stop_condition:
            drop_indices = np.random.choice(df_zero_female.index, remove_n, replace=False)
            df_zero_female.drop(drop_indices, inplace=True)

            nao_fav_nao_priv = (df_zero_female.shape[0]/(df_zero_male.shape[0] + df_zero_female.shape[0]))

        return pd.concat([df_one_male, df_one_female, df_zero_male, df_zero_female])
    elif df_name == 'Student':
        df_one_male, df_one_female = [x for _, x in df_one.groupby(df_one['sex'] == 0)]
        df_zero_male, df_zero_female = [x for _, x in df_zero.groupby(df_zero['sex'] == 0)]

        nao_fav_nao_priv = (df_zero_female.shape[0]/df_zero.shape[0])

        remove_n = int(len(df_zero_female)*0.05)
        stop_condition = 0.65
        while nao_fav_nao_priv >= stop_condition:
            drop_indices = np.random.choice(df_zero_female.index, remove_n, replace=False)
            df_zero_female.drop(drop_indices, inplace=True)

            nao_fav_nao_priv = (df_zero_female.shape[0]/(df_zero_male.shape[0] + df_zero_female.shape[0]))

        return pd.concat([df_one_male, df_one_female, df_zero_male, df_zero_female])
    else:
        df_one_male, df_one_female = [x for _, x in df_one.groupby(df_one['age'] == 0)]
        df_zero_male, df_zero_female = [x for _, x in df_zero.groupby(df_zero['age'] == 0)]

        nao_fav_priv = (df_zero_male.shape[0]/df_zero.shape[0])

        remove_n = int(len(df_zero_male)*0.05)
        stop_condition = 0.65
        while nao_fav_priv >= stop_condition:
            drop_indices = np.random.choice(df_zero_male.index, remove_n, replace=False)
            df_zero_male.drop(drop_indices, inplace=True)

            nao_fav_priv = (df_zero_male.shape[0]/(df_zero_male.shape[0] + df_zero_female.shape[0]))


        return pd.concat([df_one_male, df_one_female, df_zero_male, df_zero_female])