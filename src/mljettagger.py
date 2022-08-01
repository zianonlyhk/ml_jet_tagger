import os
from pathlib import Path
# for saving and loading the Model class
import pickle
# basic data manipulating and visualisating tools
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# importing keras and sklearn as the main modelling tools
import tensorflow as tf
from tensorflow import keras
from keras import callbacks
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score, plot_roc_curve

# outputs saved to the current directory
cwd = os.getcwd()
# directory to store output images as testing results
img_dir = cwd+'/plots/'
# directory to store saved models and history data
endpoint_dir = cwd+'/saved_models/'


class Model():

    def __init__(self, name, randomizer=777):
        self.name = name
        self.randomizer = randomizer

    def image_set(self, x_input, y_input, val_size=0.1, test_size=0.1, to_shuffle=True):
        # CNN architecture is built to take color images.
        # Therefore, input shape is: 1 colour channel + 1 imagege channel (flattened)
        x_input = x_input.reshape(
            (x_input.shape[0], x_input.shape[1], x_input.shape[2], 1))

        # Split data into training, validation and testing sets
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(x_input, y_input,
                                                                              test_size=val_size,
                                                                              shuffle=to_shuffle,
                                                                              random_state=self.randomizer)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_train, self.y_train,
                                                                                test_size=test_size /
                                                                                (1-val_size),
                                                                                shuffle=to_shuffle,
                                                                                random_state=self.randomizer)
        self.input_shape = x_input[0].shape

    def build_model(self, model=None, loss='binary_crossentropy', optimizer='adam', show_summary=False):

        # Sequential model was used for better visual interpretation from Python codes
        self.model = model
        self.model_type = str(type(self.model))

        if "keras" in self.model_type:
            # compile the training model with pre-defined parameters
            self.model.compile(loss=loss,
                               optimizer=optimizer,
                               metrics=['accuracy'])
            # print out the summary of the model if required:
            if show_summary:
                self.model.summary()
        elif "sklearn" not in self.model_type:
            raise Exception(
                "Sorry we only support keras and scikit-learn models...")

    def train_and_val(self, batch_size_para=100, nb_epochs=50):

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
            print('tagging model performance:')
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

    def save_model(self):
        try:
            Path(endpoint_dir).mkdir()
        except FileExistsError:
            pass
        Path(endpoint_dir+self.name+'.sav').touch()
        pickle.dump(self, open(endpoint_dir+self.name+'.sav', 'wb'))


def compare_models(model_list, draw_roc=True, draw_rej_eff=False):
    # 2 graphs were defined to be plotted on later
    # the ROC curve
    if draw_roc:
        fig_roc, ax_roc = plt.subplots(figsize=(7, 4))
        ax_roc.set_xlim(0, 1)
        ax_roc.set_ylim(0, 1)
    # the rejection-efficiency curve
    if draw_rej_eff:
        fig_rejeff, ax_rejeff = plt.subplots(figsize=(7, 4))
        ax_rejeff.set_yscale('log')

    # looping through the 3 tranining results and plot them together for comparison
    for each_model in model_list:

        if "keras" in each_model.model_type:
            # obtain the prediction result to pass into further calculation of roc quantities
            each_model.pred_y_test = each_model.model.predict(
                each_model.X_test)

            # a scikit-learn tool roc_curve was used to obtain information to be plotted
            fpr, tpr, _ = roc_curve(
                each_model.y_test, each_model.pred_y_test)
            # used for cutting the low value part of the rejection-efficiency curve
            len_tfpr = len(tpr)
            # obtaining the area under the curve using a scikit-learn tool
            auc = roc_auc_score(each_model.y_test, each_model.pred_y_test)

            # plot on the 2 graphs defined earlier
            # on the ROC curve
            if draw_roc:
                ax_roc.plot(fpr, tpr, label=(
                    '%s (AUC = {0:.3f})' % each_model.name).format(auc))
                ax_roc.set_xlabel('fpr')
                ax_roc.set_ylabel('tpr')
                ax_roc.set_title(
                    'ROC curve of the model(s) over the testing data set')

            # on the rejection-efficiency curve
            if draw_rej_eff:
                ax_rejeff.plot(tpr[len_tfpr//100:], 1 /
                               fpr[len_tfpr//100:], label='%s model' % each_model.name)
                ax_rejeff.set_xlabel('signal efficiency (tpr)')
                ax_rejeff.set_ylabel('Background Rejection (1/fpr)')
                ax_rejeff.set_title(
                    'Rejection/Efficiency curve of the model(s) over the testing data set')

        elif "sklearn" in each_model.model_type:
            each_model.pred_y_test = each_model.model.predict(
                each_model.X_test_vec)

            plot_roc_curve(each_model.clf, each_model.X_test_vec,
                           each_model.y_test, ax=ax_roc)
        else:
            raise Exception(
                "Sorry we only support keras and scikit-learn models...")
    if draw_roc or draw_rej_eff:
        try:
            Path(img_dir).mkdir()
        except FileExistsError:
            pass
    if draw_roc:
        ax_roc.legend()
        fig_roc.savefig(img_dir + 'roc.png')
    if draw_rej_eff:
        ax_rejeff.legend()
        fig_rejeff.savefig(img_dir + 'rejeff.png')


def load_model(name):
    return pickle.load(open(endpoint_dir+name+'.sav', 'rb'))
