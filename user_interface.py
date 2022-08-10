import numpy as np  # for loading data sets as arrays
# importing "mljettagger.py" in the ./src/ directory
import src.mljettagger as jettagger
# these modules are good for keras CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Conv2D, Activation
# these are some useful sklearn models to start from
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier

############################################# User Interface #############################################
##########################################################################################################
# define FULL DIRECTORY to the input data set, must end with a '/'
DATA_DIR = '/foo/bar/my_data/'
##########################################################################################################
##########################################################################################################

# loading raw data as images and their labels
# potential errors here if a missing '/' is present in DATA_DIR
X = np.load(DATA_DIR+'npy/x.npy')  # data
y = np.load(DATA_DIR+'npy/y.npy')  # labels

# example model 1: a keras CNN model ---------------------------------------------------------------------
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

# example model 2: a sklearn flattened model -------------------------------------------------------------
linear_model = jettagger.Model(name='test_linear_model')
linear_model.image_set(X, y)
linear_model.build_model(model=AdaBoostClassifier(),
                         show_summary=True)
linear_model.train_and_val()
linear_model.save_model()

# example loading and comparing performance --------------------------------------------------------------
test_loading1 = jettagger.load_model("test_cnn_model")
test_loading2 = jettagger.load_model("test_linear_model")
jettagger.compare_models([test_loading1, test_loading2])
