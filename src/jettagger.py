import os
import sys
# basic data manipulating and visualisating tools
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# importing keras as the main modelling tool
import tensorflow as tf
from tensorflow import keras
from keras import callbacks
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Conv2D, Activation
from tensorflow.keras.utils import to_categorical, plot_model
# some sklearn modules adapted to keras API
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score, f1_score

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
        """
        This function takes in 2 image sets for classification,
        process them and return the splitted sets ready for training

        inputs:
        X_S (numpy.ndarray), the unsplitted data set to be the primary focus of training,
        X_BG (numpy.ndarray), the unsplitted data set to be compared with the primary set,
        test_sample_size (float), the proportion of the testing set to the raw dataset,
        to_shuffle (boolean), if to further shuffle the data set, defalt True

        return ->
        X (numpy.ndarray), unsplitted dataset containing the two input image sets,
        X_train (numpy.ndarray), image dataset to be used as the traning set,
        X_test (numpy.ndarray), image dataset to be used as the testing set,
        y (numpy.ndarray), unsplitted labels of the two input image sets,
        y_train (numpy.ndarray), label dataset of the training image set,
        y_test (numpy.ndarray), label dataset of the testing image set
        """
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

    def build_model(self, architecture, loss='binary_crossentropy', optimizer='adam', show_summary=False):
        """
        This function contains the CNN model blueprint used for binary classification.
        It was optimised for W tagging performance.

        inputs:
        show_summary (boolean), to decide if the model summary would be printed

        return ->
        model (keras.engine.sequential.Sequential), the CNN model ready for fitting
        """

        # Sequential model was used for better visual interpretation from Python codes
        self.model = architecture

        # compile the training model with pre-defined parameters
        self.model.compile(loss=loss,
                           optimizer=optimizer,
                           metrics=['accuracy'])

        # print out the summary of the model if required:
        if show_summary:
            self.model.summary()

    def train_and_val(self, batch_size_para=100, nb_epochs=50):

        earlystopping = callbacks.EarlyStopping(monitor='val_loss',
                                                mode='min', patience=10,
                                                restore_best_weights=True)

        # train the model with the X and y data sets, and defined parameters
        self.history = self.model.fit(self.X_train, self.y_train,
                                      batch_size=batch_size_para,
                                      epochs=nb_epochs,
                                      validation_data=(self.X_val, self.y_val),
                                      callbacks=[earlystopping],
                                      verbose=0)

        # evaluate the W-QCD network performance
        print('tagging model performance:')
        self.model.evaluate(self.X_val, self.y_val)

    def save_model(self):
        self.model.save(endpoint_dir+self.name+'.h5')  # save the Keras model
        np.save(endpoint_dir+self.name+'_history.npy',
                self.model)  # save the training history
        # save the structure of the model as an image
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
        tf.keras.utils.plot_model(
            self.model, to_file=img_dir+self.name+'_architecture.png', show_shapes=True)

    def load_model(self, name):
        # load the Keras model
        self.model = load_model(endpoint_dir+name+'.h5')
        # load the training history
        self.history = np.load(endpoint_dir+name+'_history.npy',
                               allow_pickle='TRUE').item()

    # def evaluate_performance(self):
    #     # pyplot configuration for plotting graphs
    #     plt.rcParams['font.size'] = '12'  # set fontsize inside the plot to 12
    #     plt.rc('axes', titlesize=12)  # set fontsize of the title to 12
    #     # set fontsize of the x and y labels to 12
    #     plt.rc('axes', labelsize=12)

    #     def loss_accuracy_plot(_history, _model_label):
    #         """
    #         inputs:
    #         history (keras.callbacks.History), the model training history,
    #         model_label (str), the particle pair being classified in the network

    #         outputs:
    #         plot the training-validation-loss and training-validation-accuracy matplotlib graphs

    #         return ->
    #         null
    #         """

    #         # defined the domain as the number of epoches
    #         x_domain = np.arange(len(_history.history['loss']))

    #         # plotting the training and validation loss in the above model training
    #         fig, ax = plt.subplots(figsize=(9, 4))
    #         ax.plot(x_domain, _history.history['loss'],
    #                 label='training loss')  # training
    #         ax.plot(x_domain, _history.history['val_loss'],
    #                 label='validation loss')  # testing
    #         ax.set_xlabel('Epoch')
    #         ax.set_ylabel('loss')
    #         ax.set_title(
    #             'Comparing the training and validation loss in the %s model training process' % _model_label)
    #         ax.legend()
    #         ax.grid()
    #         plt.savefig(img_dir + 'loss_plot_%s.png' %
    #                     _model_label)  # save the plotted graph

    #         # plotting the training and validation accuracy in the above model training
    #         fig, ax = plt.subplots(figsize=(9, 4))
    #         ax.plot(x_domain, _history.history['accuracy'],
    #                 label='training accuracy')  # training
    #         ax.plot(x_domain, _history.history['val_accuracy'],
    #                 label='validation accuracy')  # testing
    #         ax.set_xlabel('Epoch')
    #         ax.set_ylabel('accuracy')
    #         ax.set_title(
    #             'Comparing the training and validation accuracy in the %s model training process' % _model_label)
    #         ax.legend()
    #         ax.grid()
    #         plt.savefig(img_dir + 'accuracy_plot_%s.png' %
    #                     _model_label)  # save the plotted graph

    #     def pred_distribution_plot(_model, _X_train, _y_train, _X_test, _y_test, _model_label, _signal, _bkg):
    #         """
    #         inputs:
    #         model (keras.engine.sequential.Sequential, in the above case), the trained model,
    #         X_train (numpy.ndarray), the splited image set for training,
    #         y_train (numpy.ndarray), the splited image set for testing,
    #         X_test (numpy.ndarray), the splited label set for training,
    #         y_test (numpy.ndarray), the splited label set for testing,
    #         model_label (str), the particle pair being classified in the network,
    #         signal (str), the jet class to be the primary focus of training,
    #         bkg (str), the jet class being compared with the primary focus

    #         outputs:
    #         plot the histogram distribution of network prediction using the training and testing sets

    #         return ->
    #         null
    #         """

    #         # separating background and signal images in the shuffled training and testing sets
    #         # Python lists are first used as it is more efficient than numpy in this case
    #         # 4 empty lists are defined, will be turned back into np array later:
    #         X_test_BG = []
    #         X_test_S = []
    #         X_train_BG = []
    #         X_train_S = []

    #         # separate on the test set
    #         for idx, indi_label in enumerate(_y_test):
    #             # then the 1st place is "0", which means the 2nd place is "1"
    #             if indi_label > 0.5:  # so it refers to "S"
    #                 X_test_S.append(_X_test[idx])
    #             else:
    #                 X_test_BG.append(_X_test[idx])

    #         # separate on the train set
    #         for idx, indi_label in enumerate(_y_train):
    #             # then the 1st place is "0", which means the 2nd place is "1"
    #             if indi_label > 0.5:  # same argument as above
    #                 X_train_S.append(_X_train[idx])
    #             else:
    #                 X_train_BG.append(_X_train[idx])

    #         # 4 numpy array were defined using the lists prepared:
    #         X_test_BG = np.array(X_test_BG)
    #         X_test_S = np.array(X_test_S)
    #         X_train_BG = np.array(X_train_BG)
    #         X_train_S = np.array(X_train_S)

    #         # predicting the results using the trained model
    #         pred_X_BG_test = _model.predict(X_test_BG)
    #         pred_X_S_test = _model.predict(X_test_S)
    #         pred_X_BG_train = _model.predict(X_train_BG)
    #         pred_X_S_train = _model.predict(X_train_S)

    #         # retreating the predicted label
    #         # the '1' place refers to the chance of having the 'signal' data characteristics
    #         pred_test_bkg = pred_X_BG_test[:]
    #         pred_test_s = pred_X_S_test[:]
    #         pred_train_bkg = pred_X_BG_train[:]
    #         pred_train_s = pred_X_S_train[:]

    #         # define the x domain as the bins of the histogram
    #         bins_array = np.linspace(0, 1, 200)

    #         # plot the 2 histogram distributions on 2 separate graphs
    #         # network confidence on the testing set
    #         fig, ax = plt.subplots(figsize=(7, 4))
    #         ax.hist(pred_test_bkg, bins=bins_array,
    #                 label='%s' % _bkg, alpha=0.5)
    #         ax.hist(pred_test_s, bins=bins_array,
    #                 label='%s' % _signal, alpha=0.5)
    #         ax.set_xlabel(
    #             'Prediction of having characteristics closer to the "signal" data')
    #         ax.set_ylabel('Histogram count')
    #         ax.set_title(
    #             'Distribution of the %s network prediction using the testing set' % _model_label)
    #         plt.legend()
    #         plt.savefig(img_dir + 'testing_dist_%s.png' % _model_label)
    #         # network confidence on the training set
    #         fig, ax = plt.subplots(figsize=(7, 4))
    #         ax.hist(pred_train_bkg, bins=bins_array,
    #                 label='%s' % _bkg, alpha=0.5)
    #         ax.hist(pred_train_s, bins=bins_array,
    #                 label='%s' % _signal, alpha=0.5)
    #         ax.set_xlabel(
    #             'Prediction of having characteristics closer to the "signal" data')
    #         ax.set_ylabel('Histogram count')
    #         ax.set_title(
    #             'Distribution of %s network prediction using the training set' % _model_label)
    #         plt.legend()
    #         plt.savefig(img_dir + 'training_dist_%s.png' % _model_label)

    #     def roc_rejeff_plot(X_test_list, y_test_list, model_list, label_list):
    #         """
    #         inputs:
    #         X_test (numpy.ndarray), the splited label set for training,
    #         y_test (numpy.ndarray), the splited label set for testing,
    #         y_test (numpy.ndarray), the splited label set for testing,
    #         y_test (numpy.ndarray), the splited label set for testing,

    #         outputs:
    #         plot the ROC curve and the rejection-efficiency curve of applying the trained network onto testing set

    #         return ->
    #         null
    #         """

    #         # 2 graphs were defined to be plotted on later
    #         # the ROC curve
    #         fig_roc, ax_roc = plt.subplots(figsize=(7, 4))
    #         ax_roc.set_xlim(0, 1)
    #         ax_roc.set_ylim(0, 1)
    #         # the rejection-efficiency curve
    #         fig_rejeff, ax_rejeff = plt.subplots(figsize=(7, 4))
    #         ax_rejeff.set_yscale('log')

    #         # looping through the 3 tranining results and plot them together for comparison
    #         for _X_test, _y_test, _model, _label in zip(X_test_list, y_test_list, model_list, label_list):

    #             # obtain the prediction result to pass into further calculation of roc quantities
    #             pred_y_test = _model.predict(_X_test)
    #             pred_y_test_column = pred_y_test[:]
    #             # a scikit-learn tool roc_curve was used to obtain information to be plotted
    #             fpr, tpr, thresholds = roc_curve(
    #                 _y_test[:], pred_y_test_column)
    #             # used for cutting the low value part of the rejection-efficiency curve
    #             len_tfpr = len(tpr)

    #             # obtaining the area under the curve using a scikit-learn tool
    #             auc = roc_auc_score(_y_test[:], pred_y_test_column)

    #             # plot on the 2 graphs defined earlier
    #             # on the ROC curve
    #             ax_roc.plot(fpr, tpr, label=(
    #                 '%s model, with AUC = {0:.3f}' % _label).format(auc))
    #             ax_roc.set_xlabel('fpr')
    #             ax_roc.set_ylabel('tpr')
    #             ax_roc.set_title(
    #                 'ROC curve of the CNN model over the testing data set')
    #             # on the rejection-efficiency curve
    #             ax_rejeff.plot(tpr[len_tfpr//100:], 1 /
    #                            fpr[len_tfpr//100:], label='%s model' % _label)
    #             ax_rejeff.set_xlabel('signal efficiency (tpr)')
    #             ax_rejeff.set_ylabel('Background Rejection (1/fpr)')
    #             ax_rejeff.set_title(
    #                 'Rejection/Efficiency curve of the CNN model over the testing data set')

    #         # label the 2 graphs
    #         ax_roc.legend()
    #         ax_rejeff.legend()
    #         # save the 2 graphs
    #         fig_roc.savefig(img_dir + 'roc.png')
    #         fig_rejeff.savefig(img_dir + 'rejeff.png')

    #         pass

    #     loss_accuracy_plot(self.history, 'electron-jet')
    #     pred_distribution_plot(self.model,
    #                            self.X_train, self.y_train, self.X_test, self.y_test,
    #                            'ele-jet', 'electron', 'jet')
    #     roc_rejeff_plot([self.X_test],
    #                     [self.X_test],
    #                     [self.model],
    #                     ['ele-jet'])
