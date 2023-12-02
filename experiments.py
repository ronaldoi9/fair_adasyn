import warnings
warnings.filterwarnings("ignore")

from datasets import get_and_prep_adult_dataset, get_and_prep_compas_dataset, get_and_prep_default_dataset, \
get_and_prep_bank_dataset, get_and_prep_student_dataset, get_and_prep_german_dataset

from tests import run_cross_validation_tests

def config_experiments_to_run_adult_dataset():

    dataset_orig = get_and_prep_adult_dataset()
    protected_attributes = ['sex', 'race']
    df_name = 'Adult'

    return dataset_orig, protected_attributes, df_name

def config_experiments_to_run_compas_dataset():

    dataset_orig = get_and_prep_compas_dataset()
    protected_attributes = ['sex', 'race']
    df_name = 'Compas'

    return dataset_orig, protected_attributes, df_name

def config_experiments_to_run_default_dataset():

    dataset_orig = get_and_prep_default_dataset()
    protected_attributes = ['sex']
    df_name = 'Default'

    return dataset_orig, protected_attributes, df_name

def config_experiments_to_run_bank_dataset():

    dataset_orig = get_and_prep_bank_dataset()
    protected_attributes = ['age']
    df_name = 'Bank'

    return dataset_orig, protected_attributes, df_name

def config_experiments_to_run_student_dataset():

    dataset_orig = get_and_prep_student_dataset()
    protected_attributes = ['sex']
    df_name = 'Student'

    return dataset_orig, protected_attributes, df_name

def config_experiments_to_run_german_dataset():

    dataset_orig = get_and_prep_german_dataset()
    protected_attributes = ['sex']
    df_name = 'German'

    return dataset_orig, protected_attributes, df_name

if __name__ == '__main__':
    RND_SEED = 44

    datasets = ['Adult', 'Bank', 'Compas', 'Default', 'German', 'Student']
    for dataset in datasets:
        if dataset == 'Adult':
            # Run Adult Dataset
            dataset_orig, protected_attributes, df_name = config_experiments_to_run_adult_dataset()
        elif dataset == 'Bank':
            # Run Bank Dataset
            dataset_orig, protected_attributes, df_name = config_experiments_to_run_bank_dataset()
        elif dataset == 'Compas':
            # Run Compas Dataset
            dataset_orig, protected_attributes, df_name = config_experiments_to_run_compas_dataset()
        elif dataset == 'Default':
            # Run Default Dataset
            dataset_orig, protected_attributes, df_name = config_experiments_to_run_default_dataset()
        elif dataset == 'German':
            # Run German Dataset
            dataset_orig, protected_attributes, df_name = config_experiments_to_run_german_dataset()
        else:
            # Run Student Dataset
            dataset_orig, protected_attributes, df_name = config_experiments_to_run_student_dataset()

        print(dataset_orig)

        ######### TEST #########
        X = dataset_orig.drop(['Probability'], axis=1)
        y = dataset_orig['Probability']

        dict_weight = {}
        for protected in protected_attributes:
            weights_norm = X[protected].value_counts(normalize=True)
            dict_weight[protected] = {}
            for label, value in weights_norm.items():
                dict_weight[protected][label] = round(1-value,2)


        algorithms = ['FairADASYN','FairSMOTE', 'ADASYN', 'SMOTE', 'Normal']
        for algorithm in algorithms:
            run_cross_validation_tests(X, y, dict_weight, algorithm, df_name, flg_oversampling_groups=True)