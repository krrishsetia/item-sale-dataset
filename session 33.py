import numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.linear_model
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sklearn as sl
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import multiprocessing

pd.options.display.max_columns = 5
pd.options.display.max_rows = 10000000

data = pd.read_csv('Dataset.csv')
temp1 = []
temp2 = []
temp3 = []
temp4 = []

def invoice(var):
    a = var.split('-')
    temp1.append(int(a[0]))
    temp2.append(int(a[1]))
    b = a[2]
    temp3.append(int(b[0:2]))
    c = b[3:9]
    d = c.split(':')
    temp4.append(int(d[0]))
    return int(d[1])

data['Invoicetime_min'] = data['InvoiceDate'].apply(invoice)
data['Invoicetime_hr'] = pd.Series(temp4)
data['InvoiceDate_day'] = pd.Series(temp3)
data['InvoiceDate_month'] = pd.Series(temp2)
data['InvoiceDate_year'] = pd.Series(temp1)

def country(var):
    if var == 'Australia':
        return 0
    elif var == 'Belgium':
        return 1
    elif var == 'Channel Islands':
        return 2
    elif var == 'Denmark':
        return 3
    elif var == 'EIRE':
        return 4
    elif var == 'France':
        return 5
    elif var == 'Germany':
        return 6
    elif var == 'Iceland':
        return 7
    elif var == 'Italy':
        return 8
    elif var == 'Japan':
        return 9
    elif var == 'Lithuania':
        return 10
    elif var == 'Netherlands':
        return 11
    elif var == 'Norway':
        return 12
    elif var == 'Poland':
        return 13
    elif var == 'Portugal':
        return 14
    elif var == 'Spain':
        return 15
    elif var == 'Switzerland':
        return 16
    elif var == 'United Kingdom':
        return 17

def num(var):
    if var == 'POST':
        return 0
    elif var == 'D':
        return 1
    elif var == 'DOT':
        return 2
    elif var == 'M':
        return 3
    elif var == 'BANK CHARGES':
        return 4
    elif var == 'S':
        return 5
    elif var == 'AMAZONFEE':
        return 6
    elif var == 'm':
        return 7
    elif var == 'DCGSSBOY':
        return 8
    elif var == 'DCGSSGIRL':
        return 9
    elif var == 'PADS':
        return 10
    elif var == 'B':
        return 11
    elif var == 'CRUK':
        return 12
    else:
        a = ''.join(filter(str.isnumeric, var))
        return int(a)

def invnum(var):
    a = ''.join(filter(str.isnumeric, var))
    return int(a)
data['Country'] = data['Country'].apply(country)
data['StockCode'] = data['StockCode'].apply(num)
data['InvoiceNo'] = data['InvoiceNo'].apply(invnum)
data.drop(columns=['Description','InvoiceDate'],axis=1,inplace=True)
data.info()
x = data['Quantity'].values.reshape(-1,1)
y = data['UnitPrice'].values.reshape(-1,1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train,y_train)
x_test_scaled = scaler.transform(x_test)

regression = LinearRegression()

regression.fit(x_train_scaled,y_train)
y_pred = regression.predict(x_test_scaled)

plt.scatter(x_test_scaled,y_test)
plt.plot(x_test_scaled,y_pred)
plt.show()
y_pred_round = np.round(y_pred)
print('mean squared error:',metrics.mean_squared_error(y_true=y_test,y_pred=y_pred))
print('regular accuracy:',metrics.accuracy_score(y_true=y_test,y_pred=y_pred_round)*100,'%')
print('balanced accuracy:',metrics.balanced_accuracy_score(y_true=y_test,y_pred=y_pred_round)*100,'%')
print('f1:',metrics.f1_score(y_true=y_test,y_pred=y_pred_round,average='weighted')*100,'%')
print('precision:',metrics.precision_score(y_true=y_test,y_pred=y_pred_round,average='weighted',zero_division=0)*100,'%')
print('kappa:',metrics.cohen_kappa_score(y1=y_test,y2=y_pred_round)*100,'%')

matrix = metrics.confusion_matrix(y_true=y_test,y_pred=y_pred_round)
display = metrics.ConfusionMatrixDisplay(matrix)
display.plot()
plt.show()
