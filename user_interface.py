import src.mljettagger as jettagger
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Conv2D, Activation

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier

data_address = '/content/drive/MyDrive/ColabNotebooks/UCLHEP_summerIntern/data_fromNaoki/'
X = np.load(data_address+'npy/x.npy')
y = np.load(data_address+'npy/y.npy')

#########################################################################################################
#########################################################################################################
cnn_model = jettagger.Model(name='test_cnn_model')
cnn_model.image_set(X, y)
cnn_model.build_model(model=Sequential([
    Conv2D(8, (3, 1), activation='relu',
           input_shape=cnn_model.input_shape, padding='same'),
    MaxPooling2D((2, 1)),
    Dropout(rate=0.2),
    Conv2D(8, (2, 2), activation='relu',
           input_shape=cnn_model.input_shape, padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(rate=0.2),
    Flatten(),
    Dense(16),
    Activation('relu'),
    Dropout(rate=0.2),
    Dense(1, activation='sigmoid')]),
    show_summary=True)
cnn_model.train_and_val()
cnn_model.save_model()
#########################################################################################################
#########################################################################################################
linear_model = jettagger.Model(name='test_linear_model')
linear_model.image_set(X, y)
linear_model.build_model(model=AdaBoostClassifier(),
                         show_summary=True)
linear_model.train_and_val()
linear_model.save_model()
#########################################################################################################
#########################################################################################################
test_loading1 = jettagger.load_model("test_cnn_model")
test_loading2 = jettagger.load_model("test_linear_model")
jettagger.compare_models([test_loading1, test_loading2])
