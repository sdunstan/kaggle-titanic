import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint


def norm(col):
    return data[col].divide(data[col].max(), fill_value=0.0)


def convert_embarked(row):
    if row['Embarked'] == 'C':
        return 0.0
    if row['Embarked'] == 'Q':
        return 0.5
    if row['Embarked'] == 'S':
        return 1.0

data = pd.read_csv('../input/train.csv')

# First, make sure all features are on the same scale. Approx -1 to 1
data['norm_sex'] = (data['Sex'] == 'female').astype(float)
data['norm_age'] = norm('Age')
data['norm_sib'] = norm('SibSp')
data['norm_parch'] = norm('Parch')
data['norm_pclass'] = norm('Pclass')
data['norm_fare'] = norm('Fare')
data['norm_embarked'] = data.apply(convert_embarked, axis=1)
data.fillna(0.0)

X_all = np.array(data[[
    'PassengerId',
    'norm_sex',
    'norm_age',
    'norm_fare',
    'norm_pclass',
    'norm_sib',
    'norm_parch',
    'norm_embarked',
    ]].values)

features = 6

X = X_all[:, 1:(features+1)]
y = to_categorical(np.array(data['Survived'].values), 2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)

print(X_train[:9])
print(y_train[:9])
print(data[:9])

model = Sequential()
model.add(Dense(128, input_dim=features, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='titanic-weights.h5', verbose=0, save_best_only=True)
model.summary()
model.fit(X_train, y_train, batch_size=50, epochs=10, callbacks=[checkpointer], validation_data=(X_test, y_test) )

print("Predicting: ")
print(X_test[:10])

prediction = model.predict(X_test, verbose=0)

correct = 0
for i in range(0, prediction.shape[0]):
    if np.argmax(prediction[i]) == np.argmax(y_test[i]):
        correct = correct + 1

accuracy = correct / prediction.shape[0]
print("\nDONE. %f" % accuracy)
