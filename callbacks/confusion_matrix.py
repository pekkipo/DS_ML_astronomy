import numpy as np
from keras.callbacks import Callback
from metrics.confusion_cell import confusion_matrix_cell

class ConfusionMatrix(Callback):
    """
    A callback used to collect confusion matrix cell values.
    Is appropriate for multi-class problems.
    """
    def __init__(self,num_labels,matrix_saver=None,val_matrix_saver=None):
        """
        Initializes the callback
        :param num_labels: The number of classes
        :param matrix_saver: Should save the confusion matrix for each epoch
        """
        super(ConfusionMatrix, self).__init__()
        self.num_labels = num_labels
        self.matrix_saver = matrix_saver
        self.val_matrix_saver = val_matrix_saver
        self.metric_prefix = 'confusion_'
        self.val_metric_prefix = 'val_confusion_'

    def on_epoch_begin(self, epoch, logs=None):
        """
        Reset confusion matrix
        :param epoch:
        :param logs:
        :return:
        """
        self.matrix = np.zeros((self.num_labels,self.num_labels))

    def on_epoch_end(self, epoch, logs=None):
        """
        Saves confusion matrix and validation confusion matrix.
        IMPORTANT NOTE: Notice that the last batch in the validation dataset may cause
        wrong statistics (since keras returns the average)
        :param epoch:
        :param logs:
        :return:
        """
        #Save the confusion matrix for training.
        if self.matrix_saver is not None:
            self.matrix_saver(self.matrix/self.matrix.sum())

        # Saves the confusion matrix for validation.
        if self.val_matrix_saver is not None:
            self.matrix = np.zeros((self.num_labels,self.num_labels))
            self.collect_values(logs,self.val_metric_prefix)
            self.val_matrix_saver(self.matrix/self.matrix.sum())

    def on_batch_end(self, batch, logs=None):
        # Collect value for each of the confusion matrix cells
        self.collect_values(logs,self.metric_prefix)

    def collect_values(self,logs,metric_prefix):
        """
        Collects all the values for the confusion matrix
        :param logs:
        :param metric_prefix:
        :return:
        """
        for i in range(self.num_labels):
            for j in range(self.num_labels):
                self.collect_value(logs,metric_prefix,i,j)

    def collect_value(self,logs,metric_prefix,i,j):
        """
        Collects from logs the value for cell (i,j) in the confusion matrix
        :param logs: Keras logs dictionary object
        :param metric_prefix: The prefix of the metric collected up to the cell number
        :param i: true class
        :param j: predicted class
        :return:
        """
        metric_name = metric_prefix + str(i*self.num_labels + j + 1)
        self.matrix[i,j] += logs[metric_name]

    def generate_metrics(self):
        """
        Creates all metrics needed for creating the confusion matrix
        :return: A list of metrics representing the confusion matrix cells
        """
        return [confusion_matrix_cell(i, j) for i in range(self.num_labels) for j in range(self.num_labels)]

