import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Importando a tabela de dados e separando a variável dependente(y) da independente(X)
dataset = pd.read_csv('student-mat.csv')
X = dataset.iloc[:, :-1].values
y= dataset.iloc[:, -1].values

#preprocessamento dos dados: transformando variaveis categoricas em variaveis numericas
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))

#Separar a tabela em treino e teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Treinar o modelo com os dados formatados
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Prever os valores da nota G3 da tabela de teste
y_pred = regressor.predict(X_test)
np.set_printoptions(precision = 0)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_pred), 1)), 1))

#R² fitness com todas as variaveis
from sklearn.metrics import r2_score
score_all_in = r2_score(y_test, y_pred)

#Preparando tabela de dados para Backward Elimination Process
import statsmodels.api as sm
X = np.append(arr = np.ones((395, 1)).astype(int), values = X, axis = 1)

#Calculando P-valor e selecionando variáveis estatisticamente significativas

X_opt = np.array(X[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]], dtype = float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#Removidas 53 variáveis, modelo pronto
X_opt = np.array(X[:, [44, 50, 56, 57, 58]], dtype = float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#idade, relação com a família, número de faltas, notas G1 e G2

#-----------------------PREVISÕES OTIMIZADAS---------------------------

#Separar a tabela OTIMIZADA em treino e teste
from sklearn.model_selection import train_test_split
X_train_opt, X_test_opt, y_train_opt, y_test_opt = train_test_split(X_opt, y, test_size = 0.2, random_state = 0)

#Treinar o modelo OTIMIZADO com os dados formatados
from sklearn.linear_model import LinearRegression
regressor_opt = LinearRegression()
regressor_opt.fit(X_train_opt, y_train_opt)

#Prever os valores OTIMIZADOS da nota G3 da tabela de teste
y_pred_opt = regressor_opt.predict(X_test_opt)
np.set_printoptions(precision = 0)
print(np.concatenate((y_pred_opt.reshape(len(y_pred_opt), 1), y_test.reshape(len(y_pred_opt), 1)), 1))

#R² fitness com todas as variaveis
score_opt = r2_score(y_test, y_pred_opt)

optimization = str(100*(score_opt - score_all_in))
print(f'otimizado em {optimization} %')
