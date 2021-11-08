import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv(r'C:\Users\kisaz\Downloads\housing.csv')



five = st.button('Первые 5 строк датафрейма')

if five:

    st.write(df.head())



genre = st.radio("Определите размер тестовой выборки",('10', '15', '20','25','30','35'))
if genre == '10':
    X_train, X_test, y_train, y_test = train_test_split(df.drop('MEDV', axis=1),
                                                        df['MEDV'],
                                                        test_size=0.1,
                                                        random_state=2100)
    st.write('Разделили данные и передали в обучение')
    regr_model = XGBRegressor()
    regr_model.fit(X_train, y_train)
    pred = regr_model.predict(X_test)
    st.write('Обучили модель, MAE = ' + str(mean_absolute_error(y_test, pred)))

elif genre == '15':
    X_train, X_test, y_train, y_test = train_test_split(df.drop('MEDV', axis=1),
                                                        df['MEDV'],
                                                        test_size=0.15,
                                                        random_state=2100)
    st.write('Разделили данные и передали в обучение')
    regr_model = XGBRegressor()
    regr_model.fit(X_train, y_train)
    pred = regr_model.predict(X_test)
    st.write('Обучили модель, MAE = ' + str(mean_absolute_error(y_test, pred)))

elif genre == '20':
    X_train, X_test, y_train, y_test = train_test_split(df.drop('MEDV', axis=1),
                                                        df['MEDV'],
                                                        test_size=0.20,
                                                        random_state=2100)
    st.write('Разделили данные и передали в обучение')
    regr_model = XGBRegressor()
    regr_model.fit(X_train, y_train)
    pred = regr_model.predict(X_test)
    st.write('Обучили модель, MAE = ' + str(mean_absolute_error(y_test, pred)))
elif genre == '25':
    X_train, X_test, y_train, y_test = train_test_split(df.drop('MEDV', axis=1),
                                                        df['MEDV'],
                                                        test_size=0.25,
                                                        random_state=2100)
    st.write('Разделили данные и передали в обучение')
    regr_model = XGBRegressor()
    regr_model.fit(X_train, y_train)
    pred = regr_model.predict(X_test)
    st.write('Обучили модель, MAE = ' + str(mean_absolute_error(y_test, pred)))
elif genre == '30':
    X_train, X_test, y_train, y_test = train_test_split(df.drop('MEDV', axis=1),
                                                        df['MEDV'],
                                                        test_size=0.30,
                                                        random_state=2100)
    st.write('Разделили данные и передали в обучение')
    regr_model = XGBRegressor()
    regr_model.fit(X_train, y_train)
    pred = regr_model.predict(X_test)
    st.write('Обучили модель, MAE = ' + str(mean_absolute_error(y_test, pred)))
elif genre == '35':
    X_train, X_test, y_train, y_test = train_test_split(df.drop('MEDV', axis=1),
                                                        df['MEDV'],
                                                        test_size=0.35,
                                                        random_state=2100)
    st.write('Разделили данные и передали в обучение')
    regr_model = XGBRegressor()
    regr_model.fit(X_train, y_train)
    pred = regr_model.predict(X_test)
    st.write('Обучили модель, MAE = ' + str(mean_absolute_error(y_test, pred)))

if st.button('График линейной регрессии'):
    fig, ax = plt.subplots(figsize=(15, 11))

    ax.plot(y_test.values, "blue", linewidth=5, label = 'Реальные значения')
    ax.plot(pred, "red", linewidth=5, label = 'Наш первый ML')
    ax.legend(loc='best')
    ax.grid(True)

    st.pyplot(fig)

if st.button('Результат предсказания'):
    st.write(pd.DataFrame({'ML':pred,'Real':y_test}))
