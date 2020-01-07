from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.constraints import unit_norm
import pickle
import numpy as np

RESOURCES = "../resources/"
train_faces, train_races, test_faces, test_races = RESOURCES + "train_embeddings.pickle", \
                                                   RESOURCES + "train_races.pickle", \
                                                   RESOURCES + "test_embeddings.pickle", RESOURCES + "test_races.pickle"

model = Sequential()
model.add(Dense(2048, activation="relu", input_shape=(2048,), kernel_constraint=unit_norm()))
model.add(Dense(4, activation="softmax"))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

with open(train_faces, "rb") as handle:
    X_train = np.squeeze(np.array(pickle.load(handle)), axis=1)
with open(train_races, "rb") as handle:
    y_train = np.array(pickle.load(handle))
with open(test_faces, "rb") as handle:
    X_test = np.squeeze(np.array(pickle.load(handle)), axis=1)
with open(test_races, "rb") as handle:
    y_test = np.array(pickle.load(handle))


y_train = to_categorical(y_train, num_classes=4)
y_test = to_categorical(y_test, num_classes=4)

model.fit(X_train, y_train, batch_size=128, epochs=15, validation_data=(X_test, y_test))
model.save(RESOURCES + "race_model.h5")
