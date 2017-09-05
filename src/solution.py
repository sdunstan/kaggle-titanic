import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import DataFrame
from keras.models import Sequential
from keras.layers.core import Dense

def feature_scaler(df, col):
    df['norm_' + col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

def sex_conversion(row):
    if row['Sex'] == 'female':
        return 1
    else:
        return 0

def convert_embarked(row):
    if row['Embarked'] == 'C':
        return 0.0
    if row['Embarked'] == 'Q':
        return 0.5
    if row['Embarked'] == 'S':
        return 1.0

data = pd.read_csv('../input/train.csv')

def preprocess(data):
    data1 = data.copy()
    data1['norm_sex'] = data1.apply(sex_conversion, axis=1)
    data2 = data1.fillna(data.median())
    data3 = data2.copy()
    feature_scaler(data3, 'Age')
    feature_scaler(data3, 'Fare')
    if 'Survived' in data3.columns:
        y = np.array(data3['Survived'].values)
    else:
        y = 0
    X = data3[['norm_sex', 'norm_Age', 'norm_Fare']].values # we have three features

    return (X, y)

X_train, y_train = preprocess(data)

print('Train data:')
print(X_train)
print(y_train)

model = Sequential()
model.add(Dense(20, input_dim=3, activation='relu'))
model.add(Dense(1, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
model.summary()
model.fit(X_train, y_train, batch_size=50, epochs=20, verbose=1)

evalData = pd.read_csv('../input/test.csv')
X_test, _ = preprocess(evalData)
prediction = model.predict(X_test, verbose=2)

print('Test Data')
print(X_test)
print(prediction[:10])

evalDataFrame = DataFrame.from_records(evalData, 'PassengerId')
evalDataFrame['Survived'] = np.argmax(prediction, axis=1)

evalDataFrame.to_csv('prediction.csv', columns=['Survived'])
