import random as rnd

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import model_selection, preprocessing
from sklearn.metrics import accuracy_score
# from sklearn.cluster import KMeans
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# import graphviz

# from sklearn import tree

# Import data
# Survived Survival (0 = No; 1 = Yes)
# Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
# name Name
# sex Sex
# age Age
# sibsp Number of Siblings/Spouses Aboard
# parch Number of Parents/Children Aboard
# ticket Ticket Number
# fare Passenger Fare (British pound)
# cabin Cabin
# embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

# df_test = pd.read_csv("test.csv", index_col=0)
df_test = pd.read_csv("test.csv", index_col=0)
df_train = pd.read_csv("train.csv", index_col=0)
combine = [df_test, df_train]

# Investigating data, findings below
print(df_train.columns.values)
print(df_train.tail())
print(df_train.info())
print('_' * 40)
print(df_test.info())
print(df_train.describe())  # describes Numerical
print(df_train.describe(include=['O']))

# Correlation: selecting features by pivoting, findings below
print(df_train[['Pclass', 'Survived']].groupby(
    ['Pclass'], as_index=True).mean().sort_values(by='Survived', ascending=False))
print(df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean())
print(df_train[['SibSp', 'Survived']].groupby(
    ['SibSp'], as_index=True).mean().sort_values(by='Survived', ascending=False))
print(df_train[['Parch', 'Survived']].groupby(
    ['Parch'], as_index=True).mean().sort_values(by='Survived', ascending=False))

# Vizualizing data, findings below
g = sns.FacetGrid(df_train, col='Survived', sharey=True)
g.map(plt.hist, 'Age', bins=20)
g = sns.FacetGrid(df_train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
g.map(plt.hist, 'Age', alpha=1, bins=20)
g.add_legend()
g = sns.FacetGrid(df_train, row='Embarked', size=2.2, aspect=1.6)
g.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
g.add_legend()
g = sns.FacetGrid(df_train, col='Survived',
                  row='Embarked', size=2.2, aspect=1.6)
g.map(sns.barplot, 'Sex', 'Fare', ci=None)
plt.show()

# Findings:
# Train data: Age, Cabin and embarked has null values
# Test data: Age, fare and cabin has null data
# Clear corr in Pclass as higher class survive more, Sex as female survive more
# and SibSp as fewer siblings/spouses seem to survive more
# Nothing super clear from Parch
# Infants had a higher survival rate and elderly a lower one
# Port of embark greatly correlates with survival rate for PClass and sex

# Dropping data we found unnecessary
# print("Before:", combine[0].shape, combine[1].shape)
# print('_' * 25, '\n')
for df in combine:
    df.drop(['Ticket', 'Cabin'], inplace=True, axis=1)
# print("After:", combine[0].shape, combine[1].shape)

# Consolidate create new titles and consolidate them
for df in combine:
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    df.drop(['Name'], axis=1, inplace=True)

# print(pd.crosstab(combine[1]['Title'], combine[1]['Sex']))

for df in combine:
    df['Title'] = df['Title'].replace(['Capt', 'Col', 'Countess', 'lady',
                                       'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                       'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

# print(combine[1][['Title', 'Survived']].groupby(
#     ['Title']).mean().sort_values(by='Survived', ascending=False))

# Creating numerical features
title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
sex_mapping = {'male': 0, 'female': 1}

for df in combine:
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'] = df['Title'].fillna(0)
    df['Sex'] = df['Sex'].map(sex_mapping).astype(int)
    # print(df.info())

# Visualize the age to decide how to fill nan
# g = sns.FacetGrid(combine[1], col='Sex', row='Pclass', size=2.2, aspect=1.6)
# g.map(plt.hist, 'Age', alpha=.5, bins=20)
# plt.show()

# Guess age depending on other features
guess_age = np.array([[0, 0, 0],
                      [0, 0, 0]])
# print(combine[1].info())
for df in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = df[(df['Sex'] == i) &
                          (df['Pclass'] == j + 1)]['Age'].dropna()
            guess = guess_df.median()
            guess_age[i, j] = int(guess)
    for i in range(0, 2):
        for j in range(0, 3):
            df.loc[(df.Age.isnull()) & (df.Sex == i) & (df.Pclass == j + 1),
                   'Age'] = guess_age[i, j]
# for df in combine:
#     df['AgeBand'] = pd.cut(df['Age'], 5)
# test_corr = combine[1][['AgeBand', 'Survived']].groupby(['AgeBand']).mean()
# print(test_corr)
for df in combine:
    df.loc[(df['Age'] <= 16), 'Age'] = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
    df.loc[(df['Age'] > 64), 'Age'] = 4
    df.Age = df.Age.astype(int)


# Create family size
for df in combine:
    df['FamilySize'] = df['SibSp'] + df['Parch']

for df in combine:
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 0, 'IsAlone'] = 1
    df.drop(['FamilySize', 'SibSp', 'Parch'], axis=1, inplace=True)

# print(combine[1][['IsAlone', 'Survived']].groupby(['IsAlone']).mean())

mode_port = combine[1]['Embarked'].dropna().mode()[0]
# print(combine[1]['Embarked'].dropna())
for df in combine:
    df['Embarked'] = df['Embarked'].fillna(mode_port)

# print(combine[1][['Embarked', 'Survived']].groupby(
#     ['Embarked']).mean().sort_values(by='Survived', ascending=False))
combine[1]['Title'] = combine[1]['Title'].astype(int)

for df in combine:
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

for df in combine:
    df['Fare'].fillna(df['Fare'].dropna().median(), inplace=True)

# combine[1]['FareBand'] = pd.qcut(df['Fare'], 4)
# print(combine[1][['FareBand', 'Survived']].groupby(['FareBand']).mean())

for df in combine:
    df.loc[df['Fare'] <= 7.91, 'Fare'] = 0
    df.loc[(df['Fare'] > 7.92) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31.0), 'Fare'] = 2
    df.loc[df['Fare'] > 31.0, 'Fare'] = 3
    df.Fare = df.Fare.astype(int)


print(df.head())

# print(combine[1].head())
# print('_' * 40)
# print(combine[1].info())

# Create new CSV files
combine[0].to_csv('modified_test.csv', index=True)
combine[1].to_csv('modified_train.csv', index=True)
