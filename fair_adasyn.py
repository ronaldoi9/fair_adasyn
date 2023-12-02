import pandas as pd
import numpy  as np
from sklearn import neighbors

seed = 44
np.random.seed(seed)

def update_dict_weight(X, y, columns, dict_weight, syn_data, label_minority):
    data = []
    for values in syn_data:
        data.append(values[0])

    # Concatenate the positive labels with the newly made data
    labels = np.full((len(data), 1), label_minority)
    data = np.concatenate([data, labels], axis=1)

    # Concatenate with old data
    org_data = np.concatenate([X, y.reshape(-1, 1)], axis=1)
    data = np.concatenate([data, org_data])

    # Convert to dataframe
    all_columns = np.append(columns, 'Probability')
    df_over = pd.DataFrame(data, columns=all_columns)

    # Get the protected attributes from Dict Weight
    protected_attributes = list(dict_weight.keys())

    X_over = df_over.drop(['Probability'], axis=1)
    new_dict_weight = {}
    for protected in protected_attributes:
        weights_norm = X_over[protected].value_counts(normalize=True)
        new_dict_weight[protected] = {}
        for label, value in weights_norm.items():
            new_dict_weight[protected][label] = round(1-value,2)

    return new_dict_weight


def fair_adasyn(X, y, dict_weight, beta, K, threshold=1, flg_update_dict=True):

    """
    Adaptively generating minority data samples according to their distributions, and aims to bring more fairness on generate synthetic data proccess.
    More synthetic data is generated for minority class samples that are harder to learn.
    Harder to learn data is defined as positive examples with not many examples for in their respective neighbourhood.

               Inputs
                -----
                   X:  Input features, X, sorted by the minority examples on top.  Minority example should also be labeled as 1
                   y:  Labels, with minority example labeled as 1
         dict_weight:  Dictionary that contains the probability of creating a protected attribute
                beta:  Degree of imbalance desired. 
                   K:  Amount of neighbours to look at
           threshold:  Amount of imbalance rebalance required for algorithm
     flg_update_dict:  Flag that indicate if needs update dictionary weights

          Variables
              -----
                xi:  Minority example
                xzi:  A minority example inside the neighbourhood of xi
                ms:  Amount of data in minority class
                ml:  Amount of data in majority class
                clf:  k-NN classifier model
                  d:  Ratio of minority : majority
              beta:  Degree of imbalance desired
                  G:  Amount of data to generate
                Ri:  Ratio of majority data / neighbourhood size.  Larger ratio means the neighbourhood is harder to learn,
                      thus generating more data.
            Minority_per_xi:  All the minority data's index by neighbourhood
            Rhat_i:  Normalized Ri, where sum = 1
                Gi:  Amount of data to generate per neighbourhood (indexed by neighbourhoods corresponding to xi)

          Returns
              -----
  syn_data:  New synthetic minority data created
    """

    label_minority = y.value_counts().idxmin()
    ms = len(np.where(y == label_minority)[0])

    ml = X.shape[0] - ms

    print('ms: ', ms)
    print('ml: ', ml)

    columns = X.columns
    X = X.values

    clf = neighbors.KNeighborsClassifier()
    clf.fit(X, y)

    # Step 1, calculate the degree of class imbalance.  If degree of class imbalance is violated, continue.
    d = np.divide(ms, ml)
    print('d: ', d)

    if d > threshold:
        return print("The data set is not imbalanced enough.")

    # Step 2a, if the minority data set is below the maximum tolerated threshold, generate data.
    # Beta is the desired balance level parameter.  Beta > 1 means u want more of the imbalanced type, vice versa.
    G = (ml - ms) * beta
    print('G: ', G)

    label_minority = y.value_counts().idxmin()
    idxs_minority = np.where(y == label_minority)[0]

    # Convert y to array
    y = y.values

    Ri = []
    Minority_per_xi = []
    for idx in idxs_minority:
        xi = X[idx].reshape(1, -1)
        # Returns indices of the closest neighbours, and return it as a list
        neighbours = clf.kneighbors(xi, n_neighbors=K+1, return_distance=False)[:, 1:][0]

        # Count how many belongs to the majority class
        count = 0
        for idx_ngbr in neighbours:
            if y[idx_ngbr] != label_minority:
                count += 1

        Ri.append(round(count / K, 3))

        # Find all the minority examples
        minority = []
        for idx_ngbr in neighbours:
             if y[idx_ngbr] == label_minority:
                minority.append(idx_ngbr)

        Minority_per_xi.append(minority)

    # Step 2c, normalize ri's so their sum equals to 1
    Rhat_i = []
    for ri in Ri:
        rhat_i = ri / sum(Ri)
        Rhat_i.append(rhat_i)

    assert(sum(Rhat_i) > 0.99)

    # Step 2d, calculate the number of synthetic data examples that will be generated for each minority example
    Gi = []
    for rhat_i in Rhat_i:
        gi = np.ceil(rhat_i * G).astype(int)
        Gi.append(gi)

    # Set intervals to update dict weight
    total_syn_data = sum(Gi)
    dict_update_range = int(total_syn_data/10)

    # # Step 2e, generate synthetic examples
    syn_data = []
    for i, idx in enumerate(idxs_minority):
        xi = X[idx].reshape(1, -1)
        for _ in range(Gi[i]):
            # If the minority list is not empty
            if Minority_per_xi[i]:
                syn_data_i = []
                index = np.random.choice(Minority_per_xi[i])
                xzi = X[index].reshape(1, -1)
                for i, _ in enumerate(xzi[0]):
                    if columns[i] in dict_weight.keys(): 
                      syn_data_i.append( np.random.choice(list(dict_weight[columns[i]].keys()), 1, p=list(dict_weight[columns[i]].values()))[0] )
                    else:
                      syn_data_i.append( xi[0][i] + (xzi[0][i] - xi[0][i]) * np.random.uniform(0, 1) )
                syn_data.append(np.array([syn_data_i]))
                if (len(syn_data) > dict_update_range) and flg_update_dict:
                    print('dict_weight: ', dict_weight)
                    dict_weight = update_dict_weight(X, y, columns, dict_weight, syn_data, label_minority)
                    dict_update_range += dict_update_range


    # Build the data matrix
    data = []
    for values in syn_data:
        data.append(values[0])

    print("{} amount of minority class samples generated".format(len(data)))

    if len(syn_data) == 0:
        print('Fail - All minority samples are outliers')
        org_data = np.concatenate([X, y.reshape(-1, 1)], axis=1)

        # Convert to dataframe
        all_columns = np.append(columns, 'Probability')
        df = pd.DataFrame(org_data, columns=all_columns)

        # Return x data and y data
        X_orig = df.drop(['Probability'], axis=1)
        y_orig = df['Probability']

        return X_orig, y_orig

    # Concatenate the positive labels with the newly made data
    labels = np.full((len(data), 1), label_minority)
    data = np.concatenate([data, labels], axis=1)

    # Concatenate with old data
    org_data = np.concatenate([X, y.reshape(-1, 1)], axis=1)
    data = np.concatenate([data, org_data])

    # Convert to dataframe
    all_columns = np.append(columns, 'Probability')
    df_syn = pd.DataFrame(data, columns=all_columns)

    # Return x data and y data
    X_res = df_syn.drop(['Probability'], axis=1)
    y_res = df_syn['Probability']

    return X_res, y_res