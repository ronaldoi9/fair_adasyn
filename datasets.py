import pandas as pd
import numpy as np

DATA_LOCAL = 'data'

def get_and_prep_adult_dataset():
    dataset_orig = pd.read_csv(f'{DATA_LOCAL}/adult.data.csv')

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()

    ## Drop categorical features
    dataset_orig = dataset_orig.drop(['workclass','fnlwgt','education','marital-status','occupation','relationship','native-country'],axis=1)

    ## Change symbolics to numerics
    dataset_orig['sex'] = np.where(dataset_orig['sex'] == ' Male', 1, 0)
    dataset_orig['race'] = np.where(dataset_orig['race'] != ' White', 0, 1)
    dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == ' <=50K', 0, 1)

    ## Discretize age
    dataset_orig['age'] = np.where(dataset_orig['age'] >= 70, 70, dataset_orig['age'])
    dataset_orig['age'] = np.where((dataset_orig['age'] >= 60 ) & (dataset_orig['age'] < 70), 60, dataset_orig['age'])
    dataset_orig['age'] = np.where((dataset_orig['age'] >= 50 ) & (dataset_orig['age'] < 60), 50, dataset_orig['age'])
    dataset_orig['age'] = np.where((dataset_orig['age'] >= 40 ) & (dataset_orig['age'] < 50), 40, dataset_orig['age'])
    dataset_orig['age'] = np.where((dataset_orig['age'] >= 30 ) & (dataset_orig['age'] < 40), 30, dataset_orig['age'])
    dataset_orig['age'] = np.where((dataset_orig['age'] >= 20 ) & (dataset_orig['age'] < 30), 20, dataset_orig['age'])
    dataset_orig['age'] = np.where((dataset_orig['age'] >= 10 ) & (dataset_orig['age'] < 10), 10, dataset_orig['age'])
    dataset_orig['age'] = np.where(dataset_orig['age'] < 10, 0, dataset_orig['age'])

    return dataset_orig

def get_and_prep_compas_dataset():
    ## Load dataset
    dataset_orig = pd.read_csv(f'{DATA_LOCAL}/compas-scores-two-years.csv')

    ## Drop categorical features
    ## Removed two duplicate coumns - 'decile_score','priors_count'
    dataset_orig = dataset_orig.drop(['id','name','first','last','compas_screening_date',
                                    'dob','age','juv_fel_count','decile_score',
                                    'juv_misd_count','juv_other_count','days_b_screening_arrest',
                                    'c_jail_in','c_jail_out','c_case_number','c_offense_date','c_arrest_date',
                                    'c_days_from_compas','c_charge_desc','is_recid','r_case_number','r_charge_degree',
                                    'r_days_from_arrest','r_offense_date','r_charge_desc','r_jail_in','r_jail_out',
                                    'violent_recid','is_violent_recid','vr_case_number','vr_charge_degree','vr_offense_date',
                                    'vr_charge_desc','type_of_assessment','decile_score','score_text','screening_date',
                                    'v_type_of_assessment','v_decile_score','v_score_text','v_screening_date','in_custody',
                                    'out_custody','start','end','event'],axis=1)

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()

    ## Change symbolics to numerics
    dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'Male', 1, 0)
    dataset_orig['race'] = np.where(dataset_orig['race'] == 'Caucasian', 1, 0)
    dataset_orig['priors_count'] = np.where((dataset_orig['priors_count'] >= 1 ) & (dataset_orig['priors_count'] <= 3), 3, dataset_orig['priors_count'])
    dataset_orig['priors_count'] = np.where(dataset_orig['priors_count'] > 3, 4, dataset_orig['priors_count'])
    dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == 'Greater than 45',45,dataset_orig['age_cat'])
    dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == '25 - 45', 25, dataset_orig['age_cat'])
    dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == 'Less than 25', 0, dataset_orig['age_cat'])
    dataset_orig['c_charge_degree'] = np.where(dataset_orig['c_charge_degree'] == 'F', 1, 0)

    ## Rename class column
    dataset_orig.rename(index=str, columns={"two_year_recid": "Probability"}, inplace=True)

    #dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 0, 1, 0)

    return dataset_orig

def get_and_prep_german_dataset():
    dataset_orig = pd.read_csv(f'{DATA_LOCAL}/GermanData.csv')

    ## Drop categorical features
    dataset_orig = dataset_orig.drop(['1','2','4','5','8','10','11','12','14','15','16','17','18','19','20'],axis=1)

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()

    ## Change symbolics to numerics
    dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A91', 1, dataset_orig['sex'])
    dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A92', 0, dataset_orig['sex'])
    dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A93', 1, dataset_orig['sex'])
    dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A94', 1, dataset_orig['sex'])
    dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A95', 0, dataset_orig['sex'])


    dataset_orig['age'] = np.where(dataset_orig['age'] >= 25, 1, 0)
    dataset_orig['credit_history'] = np.where(dataset_orig['credit_history'] == 'A30', 1, dataset_orig['credit_history'])
    dataset_orig['credit_history'] = np.where(dataset_orig['credit_history'] == 'A31', 1, dataset_orig['credit_history'])
    dataset_orig['credit_history'] = np.where(dataset_orig['credit_history'] == 'A32', 1, dataset_orig['credit_history'])
    dataset_orig['credit_history'] = np.where(dataset_orig['credit_history'] == 'A33', 2, dataset_orig['credit_history'])
    dataset_orig['credit_history'] = np.where(dataset_orig['credit_history'] == 'A34', 3, dataset_orig['credit_history'])

    dataset_orig['savings'] = np.where(dataset_orig['savings'] == 'A61', 1, dataset_orig['savings'])
    dataset_orig['savings'] = np.where(dataset_orig['savings'] == 'A62', 1, dataset_orig['savings'])
    dataset_orig['savings'] = np.where(dataset_orig['savings'] == 'A63', 2, dataset_orig['savings'])
    dataset_orig['savings'] = np.where(dataset_orig['savings'] == 'A64', 2, dataset_orig['savings'])
    dataset_orig['savings'] = np.where(dataset_orig['savings'] == 'A65', 3, dataset_orig['savings'])

    dataset_orig['employment'] = np.where(dataset_orig['employment'] == 'A72', 1, dataset_orig['employment'])
    dataset_orig['employment'] = np.where(dataset_orig['employment'] == 'A73', 1, dataset_orig['employment'])
    dataset_orig['employment'] = np.where(dataset_orig['employment'] == 'A74', 2, dataset_orig['employment'])
    dataset_orig['employment'] = np.where(dataset_orig['employment'] == 'A75', 2, dataset_orig['employment'])
    dataset_orig['employment'] = np.where(dataset_orig['employment'] == 'A71', 3, dataset_orig['employment'])

    ## ADD Columns
    dataset_orig['credit_history=Delay'] = 0
    dataset_orig['credit_history=None/Paid'] = 0
    dataset_orig['credit_history=Other'] = 0

    dataset_orig['credit_history=Delay'] = np.where(dataset_orig['credit_history'] == 1, 1, dataset_orig['credit_history=Delay'])
    dataset_orig['credit_history=None/Paid'] = np.where(dataset_orig['credit_history'] == 2, 1, dataset_orig['credit_history=None/Paid'])
    dataset_orig['credit_history=Other'] = np.where(dataset_orig['credit_history'] == 3, 1, dataset_orig['credit_history=Other'])

    dataset_orig['savings=500+'] = 0
    dataset_orig['savings=<500'] = 0
    dataset_orig['savings=Unknown/None'] = 0

    dataset_orig['savings=500+'] = np.where(dataset_orig['savings'] == 1, 1, dataset_orig['savings=500+'])
    dataset_orig['savings=<500'] = np.where(dataset_orig['savings'] == 2, 1, dataset_orig['savings=<500'])
    dataset_orig['savings=Unknown/None'] = np.where(dataset_orig['savings'] == 3, 1, dataset_orig['savings=Unknown/None'])

    dataset_orig['employment=1-4 years'] = 0
    dataset_orig['employment=4+ years'] = 0
    dataset_orig['employment=Unemployed'] = 0

    dataset_orig['employment=1-4 years'] = np.where(dataset_orig['employment'] == 1, 1, dataset_orig['employment=1-4 years'])
    dataset_orig['employment=4+ years'] = np.where(dataset_orig['employment'] == 2, 1, dataset_orig['employment=4+ years'])
    dataset_orig['employment=Unemployed'] = np.where(dataset_orig['employment'] == 3, 1, dataset_orig['employment=Unemployed'])


    dataset_orig = dataset_orig.drop(['credit_history','savings','employment'],axis=1)
    ## In dataset 1 means good, 2 means bad for probability. I change 2 to 0
    dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 2, 0, 1)

    return dataset_orig

def get_and_prep_default_dataset():
    ## Load dataset
    dataset_orig = pd.read_csv(f'{DATA_LOCAL}/default_of_credit_card_clients.csv', skiprows=[0])
    dataset_orig.columns = dataset_orig.columns.str.lower()

    ## Change column values
    # 1 male, 2 female
    dataset_orig['sex'] = np.where(dataset_orig['sex'] == 2, 0,1)

    # Remove id columns
    dataset_orig.drop(['id'], axis=1, inplace=True)

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()

    dataset_orig.rename(index=str, columns={"default payment next month": "Probability"}, inplace=True)

    return dataset_orig

def get_and_prep_bank_dataset():
    dataset_orig = pd.read_csv(f'{DATA_LOCAL}/bank.csv', sep=';')

    # Drop categorical features
    dataset_orig = dataset_orig.drop(['job','marital','education','contact','month','poutcome'],axis=1)

    dataset_orig['default'] = np.where(dataset_orig['default'] == 'no', 0, 1)
    dataset_orig['housing'] = np.where(dataset_orig['housing'] == 'no', 0, 1)
    dataset_orig['loan'] = np.where(dataset_orig['loan'] == 'no', 0, 1)
    dataset_orig['Probability'] = np.where(dataset_orig['y'] == 'yes', 1, 0)

    dataset_orig.drop(['y'], axis=1, inplace=True)
    dataset_orig['age'] = np.where(dataset_orig['age'] >= 30, 1, 0)

    return dataset_orig

def get_and_prep_student_dataset():
    ## Load dataset
    dataset_orig = pd.read_csv(f'{DATA_LOCAL}/Student.csv')

    ## Drop NULL values
    dataset_orig = dataset_orig.dropna()

    ## Drop categorical features
    dataset_orig = dataset_orig.drop(['school','address', 'famsize', 'Pstatus','Mjob', 'Fjob', 'reason', 'guardian'],axis=1)

    ## Change symbolics to numerics
    dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'M', 1, 0)
    dataset_orig['schoolsup'] = np.where(dataset_orig['schoolsup'] == 'yes', 1, 0)
    dataset_orig['famsup'] = np.where(dataset_orig['famsup'] == 'yes', 1, 0)
    dataset_orig['paid'] = np.where(dataset_orig['paid'] == 'yes', 1, 0)
    dataset_orig['activities'] = np.where(dataset_orig['activities'] == 'yes', 1, 0)
    dataset_orig['nursery'] = np.where(dataset_orig['nursery'] == 'yes', 1, 0)
    dataset_orig['higher'] = np.where(dataset_orig['higher'] == 'yes', 1, 0)
    dataset_orig['internet'] = np.where(dataset_orig['internet'] == 'yes', 1, 0)
    dataset_orig['romantic'] = np.where(dataset_orig['romantic'] == 'yes', 1, 0)
    dataset_orig['Probability'] = np.where(dataset_orig['Probability'] > 12, 1, 0)

    return dataset_orig