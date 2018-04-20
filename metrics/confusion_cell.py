import keras.backend as K
import tensorflow as tf

def confusion_matrix_cell(true_class, pred_class):
    """
    Creates a metric that collects the value for a single cells of the confusion matrix
    :param true_class:
    :param pred_class:
    :return: Keras metric
    """

    def confusion(y_true, y_pred):
        """
        Collects the samples predicted as pred_class where its true class is true_class
        :param y_true:
        :param y_pred:
        :return: the number of predictions as mentioned above
        """
        # Calculate the label from one-hot encoding
        pred_class_label = K.argmax(y_pred, axis=-1)
        true_class_label = K.argmax(y_true, axis=-1)

        # Create a mask representing where the prediction is pred_class and the true class is true_class
        pred_mask = K.equal(pred_class_label, pred_class)
        true_mask = K.equal(true_class_label, true_class)
        mask = tf.logical_and(pred_mask, true_mask)

        # Get the total number of occurences
        occurrences = K.sum(K.cast(mask, 'int32'), axis=0)
        return occurrences

    return confusion