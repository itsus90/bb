'''
TOPIC CHOSEN: Predict the percentage of an student based on the no. of study
hours.
DATA SET LINK: https://www.kaggle.com/code/ameythakur20/tsfinternshiptask-
1-supervised-learning
DATASET NAME: scores.csv
'''
#supervised

%matplotlib inline import
pandas as pd import numpy
as np
import matplotlib.pyplot as plt
import seaborn as sns sns.set()
plt.style.use('ggplot') data =
pd.read_csv("scores.csv") X =
data.iloc[:, :-1].values y =
data.iloc[:, 1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2,
random_state= 0)

sns.distplot(y_train, kde=True, color='green',)
plt.title('Distribution of Scores') plt.xlabel('Hours
Studied')
plt.ylabel('Percentage Scored')
                                               
sns.regplot(X_train, y_train, color='green', )
plt.title('Hours vs Scores') plt.xlabel('Hours Studied')
plt.ylabel('Percentage Scored')
                                               
#unsupervised
data.plot(x='Hours', y='Scores', color='green', style='*')
plt.title('Hours vs Percentage')
 plt.xlabel('Hours
Studied')
 plt.ylabel('Percentage Scored')
 plt.show()
