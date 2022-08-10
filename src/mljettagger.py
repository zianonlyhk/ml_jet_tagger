# for pwd, cd and touch commands
import os
from pathlib import Path
# for saving and loading the Model class
import pickle
# basic data manipulating and visualisating tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# importing keras and sklearn as the main modelling tools
import tensorflow as tf
from keras import callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, plot_roc_curve
# for including information about time when logging
from datetime import datetime, date, time
import warnings  # for tf warnings

# ignore warnings thrown by TensorFlow for a better user interface
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore')

# outputs saved to the current directory
cwd = os.getcwd()
results_dir = cwd+'/results/temp/'


class Model():

    def __init__(self, name, randomizer=777):
        self.name = name  # unique identifier
        # constant randomness for consistent comparison on splitted test set
        self.randomizer = randomizer

    def image_set(self, x_input, y_input, val_size=0.1, test_size=0.1, to_shuffle=True):
        # CNN architecture is built to take color images.
        # Therefore, input images are reshaped to have an extra dimention as inputs are monochromatic
        x_input = x_input.reshape(
            (x_input.shape[0], x_input.shape[1], x_input.shape[2], 1))

        # Split data into training, validation and testing sets
        # default ratio --> training:validation:testing = 8:1:1
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(x_input, y_input,
                                                                              test_size=val_size,
                                                                              shuffle=to_shuffle,
                                                                              random_state=self.randomizer)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_train, self.y_train,
                                                                                test_size=test_size /
                                                                                (1-val_size),
                                                                                shuffle=to_shuffle,
                                                                                random_state=self.randomizer)
        # record shape of img for flattening later
        self.input_shape = x_input[0].shape

    def build_model(self, model=None, loss='binary_crossentropy', optimizer='adam', show_summary=False):

        # pass the input model to the class and see its type
        self.model = model
        self.model_type = str(type(self.model))

        if "keras" in self.model_type:  # if keras model was received as parameter
            # compile the training model with pre-defined parameters
            self.model.compile(loss=loss,
                               optimizer=optimizer,
                               metrics=['accuracy'])
            # print out the summary of the model if required:
            if show_summary:
                self.model.summary()
        elif "sklearn" not in self.model_type:  # if sklearn model then do nothing
            raise Exception(
                "Sorry we only support keras and scikit-learn models...")

    def train_and_val(self, batch_size_para=100, nb_epochs=50):

        # define the rules of early stopping to save traninig time
        earlystopping = callbacks.EarlyStopping(monitor='val_loss',
                                                mode='min', patience=10,
                                                restore_best_weights=True)

        if "keras" in self.model_type:
            # train the model with the X and y data sets, and defined parameters
            self.history = self.model.fit(self.X_train, self.y_train,
                                          batch_size=batch_size_para,
                                          epochs=nb_epochs,
                                          validation_data=(
                                              self.X_val, self.y_val),
                                          callbacks=[earlystopping],
                                          verbose=0)
            # evaluate the W-QCD network performance
            print('\ntagging model performance:')
            self.model.evaluate(self.X_val, self.y_val)
            self.pred_y_test = None
        elif "sklearn" in self.model_type:
            self.X_train_vec = self.X_train.reshape(
                (self.X_train.shape[0], self.X_train.shape[1]*self.X_train.shape[2]))
            self.X_test_vec = self.X_test.reshape(
                (self.X_test.shape[0], self.X_test.shape[1]*self.X_test.shape[2]))
            self.clf = self.model.fit(self.X_train_vec, self.y_train)
            self.pred_y_test = self.clf.predict(self.X_test_vec)
        else:
            raise Exception(
                "Sorry we only support keras and scikit-learn models...")

    def save_model(self, dir=results_dir):  # save the class instance using pickle
        Path(dir+self.name+'.sav').touch()
        pickle.dump(self, open(dir+self.name+'.sav', 'wb'))


def compare_models(model_list, dir=results_dir):
    # obtain time information for the name of output img
    now = datetime.now()
    nowdate = date.today()
    nowtime = time(now.hour, now.minute)
    date_and_time = str(nowdate)+"_"+str(nowtime)

    # an roc graph is defined to be plotted on later
    fig_roc, ax_roc = plt.subplots(figsize=(7, 4))
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)

    # looping through the tranining results and plot them together for comparison
    for each_model in model_list:

        if "keras" in each_model.model_type:
            # obtain the prediction result to pass into further calculation of roc quantities
            each_model.pred_y_test = each_model.model.predict(
                each_model.X_test)

            # a scikit-learn tool roc_curve was used to obtain information to be plotted
            fpr, tpr, _ = roc_curve(
                each_model.y_test, each_model.pred_y_test)
            # obtaining the area under the curve using a scikit-learn tool
            auc = roc_auc_score(each_model.y_test, each_model.pred_y_test)

            # plot on the roc graph defined earlier
            ax_roc.plot(fpr, tpr, label=(
                '%s (AUC = {0:.3f})' % each_model.name).format(auc))
            ax_roc.set_xlabel('fpr')
            ax_roc.set_ylabel('tpr')
            ax_roc.set_title(
                'ROC curve of the model(s) over the testing data set')

        elif "sklearn" in each_model.model_type:
            # obtain the testing results to be plotted
            each_model.pred_y_test = each_model.model.predict(
                each_model.X_test_vec)
            # a different method is used to plot a smooth graph
            # the same method as above would yield a zigzag line
            plot_roc_curve(each_model.clf, each_model.X_test_vec,
                           each_model.y_test, ax=ax_roc)
        else:
            raise Exception(
                "Sorry we only support keras and scikit-learn models...")

    # see if there exsists a directory to save plot
    try:
        Path(dir).mkdir()
    except FileExistsError:
        pass

    ax_roc.legend()
    fig_roc.savefig(dir + date_and_time+'_roc.png')


def load_model(name, dir=results_dir):
    # load the whole class instance using pickle, identified by their name
    return pickle.load(open(dir+name+'.sav', 'rb'))
