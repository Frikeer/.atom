import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import model_selection, preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

# Import data
'''
survival Survival (0 = No; 1 = Yes)
Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
'''


df_test = pd.read_csv("test.csv",
                      index_col=0)
df_train = pd.read_csv("train.csv",
                       index_col=0)
df_gender = pd.read_csv("gender_submission.csv",
                        index_col=0)


# Convert and normalize the training data
df_train.drop(['Name', 'Ticket'],
              axis=1,
              inplace=True)
df_train.fillna(0, inplace=True)
# print(df_train.head())
# df_train = df_train.apply(pd.to_numeric)


def is_num(col):
    """
    Checks if a column is numerical
    """
    if col.dtype != np.int64 and col.dtype != np.float64:
        return "Categorical"
    return "Numerical"


# Creates list with type of variable
def list_types(df):
    columns = df.columns.values
    print(columns)
    types = []
    for col in columns:
        types.append(is_num(df[col]))
    return types

def turn_numerical(df,
                   type_list):
    """
    Turns dataframe into numbers
    returns new df
    """
    pieces = []
    types = []
    for col in range(0, len(type_list)):
        if type_list[col] == "Categorical":
            "zero"
            types.append(type_list[col])
        else:
            pieces.append(df.iloc[:, col].values)
            types.append(type_list[col])
    print(types)
    return pieces

types = list_types(df_train)
pieces = turn_numerical(df_train, types)
# print(pieces)
