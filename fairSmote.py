import random
import pandas as pd
from scipy.spatial import distance as dist
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors as NN

def get_ngbr(df, knn):
            rand_sample_idx = random.randint(0, df.shape[0] - 1)
            parent_candidate = df.iloc[rand_sample_idx]
            ngbr = knn.kneighbors(parent_candidate.values.reshape(1,-1),3,return_distance=False)
            candidate_1 = df.iloc[ngbr[0][0]]
            candidate_2 = df.iloc[ngbr[0][1]]
            candidate_3 = df.iloc[ngbr[0][2]]
            return parent_candidate,candidate_2,candidate_3

def generate_samples(no_of_samples,df,df_name):
    
    total_data = df.values.tolist()
    knn = NN(n_neighbors=5,algorithm='auto').fit(df)
    
    for _ in range(no_of_samples):
        cr = 0.8
        f = 0.8
        parent_candidate, child_candidate_1, child_candidate_2 = get_ngbr(df, knn)
        new_candidate = []
        for key, _ in parent_candidate.items():
            if isinstance(parent_candidate[key], bool):
                new_candidate.append(parent_candidate[key] if cr < random.random() else not parent_candidate[key])
            elif isinstance(parent_candidate[key], str):
                new_candidate.append(random.choice([parent_candidate[key],child_candidate_1[key],child_candidate_2[key]]))
            elif isinstance(parent_candidate[key], list):
                temp_lst = []
                for i, _ in enumerate(parent_candidate[key]):
                    temp_lst.append(parent_candidate[key][i] if cr < random.random() else
                                    int(parent_candidate[key][i] +
                                        f * (child_candidate_1[key][i] - child_candidate_2[key][i])))
                new_candidate.append(temp_lst)
            else:
                new_candidate.append(abs(parent_candidate[key] + f * (child_candidate_1[key] - child_candidate_2[key])))        
        total_data.append(new_candidate)
        
    final_df = pd.DataFrame(total_data)
    if df_name == 'Adult':
        final_df = final_df.rename(columns={0:"age",1:"education-num",2:"race",3:"sex",4:"capital-gain",5:"capital-loss",6:"hours-per-week",7:"Probability"}, errors="raise")
    if df_name == 'Compas':
        final_df = final_df.rename(columns={0:"sex",1:"age_cat",2:"race",3:"priors_count",4:"c_charge_degree",5:"Probability"}, errors="raise")
    if df_name == 'Default':
    	final_df = final_df.rename(columns={0: 'limit_bal', 1: 'sex', 2: 'education', 3: 'marriage', 4: 'age', 5: 'pay_0', 6: 'pay_2',7: 'pay_3', 8: 'pay_4', 9: 'pay_5', 10: 'pay_6', 11: 'bill_amt1', 12: 'bill_amt2',13: 'bill_amt3', 14: 'bill_amt4', 15: 'bill_amt5', 16: 'bill_amt6', 17: 'pay_amt1',18: 'pay_amt2', 19: 'pay_amt3', 20: 'pay_amt4', 21: 'pay_amt5', 22: 'pay_amt6',23: 'Probability'}, errors="raise")
    if df_name == 'German':
    	final_df = final_df.rename(columns={0:"sex",1:"age",2:"Probability",3:"credit_history=Delay",4:"credit_history=None/Paid",5:"credit_history=Other",6:"savings=500+",7:"savings=<500",8:"savings=Unknown/None",9:"employment=1-4 years",10:"employment=4+ years",11:"employment=Unemployed"}, errors="raise")
    if df_name == 'Heart':
    	final_df = final_df.rename(columns={0:"age",1:"sex",2:"cp",3:"trestbps",4:"chol",5:"fbs",6:"restecg",7:"thalach",8:"exang",9:"oldpeak",10:"slope",11:"ca",12:"thal",13:"Probability"}, errors="raise")
    if df_name == 'Bank':
        final_df = final_df.rename(columns={0:"age",1:"default",2:"balance",3:"housing",4:"loan",5:"day",6:"duration",7:"campaign",8:"pdays",9:"previous",10:"Probability"}, errors="raise")
    if df_name == 'Titanic':
        final_df = final_df.rename(columns={0:"Pclass",1:"sex",2:"Age",3:"SibSp",4:"Parch",5:"Fare",6:"Probability"}, errors="raise")
    if df_name == 'Student':
        final_df = final_df.rename(columns={0:'sex', 1:'age', 2:'Medu', 3:'Fedu', 4:'traveltime', 5:'studytime', 6:'failures',
       7:'schoolsup', 8:'famsup', 9:'paid', 10:'activities', 11:'nursery', 12:'higher',
       13:'internet', 14:'romantic', 15:'famrel', 16:'freetime', 17:'goout', 18:'Dalc', 19:'Walc',
       20:'health', 21:'absences', 22:'G1', 23:'G2', 24:'Probability'}, errors="raise")

    return final_df