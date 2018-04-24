'''

Classify galaxies into three types (ellipticals, spirals or galactic mergers)

Crowd-classified classes from Galaxy Zoo as the training data for our automatic decision tree classifier.

The features are: colour index, adaptive moments, eccentricities and concentrations.
These features are provided as part of the SDSS catalogue.


Colour indices are the same colours (u-g, g-r, r-i, and i-z) used for regression. Studies of galaxy evolution tell us
that spiral galaxies have younger star populations and therefore are 'bluer' (brighter at lower wavelengths).
Elliptical galaxies have an older star population and are brighter at higher wavelengths ('redder').

Eccentricity approximates the shape of the galaxy by fitting an ellipse to its profile. Eccentricity is the ratio of the two axis (semi-major and semi-minor).
The De Vaucouleurs model was used to attain these two axis. To simplify our experiments the median eccentricity across the 5 filters will be used

Adaptive moments also describe the shape of a galaxy. They are used in image analysis to detect similar objects
at different sizes and orientations. Use the fourth moment here for each band.

Concentration is similar to the luminosity profile of the galaxy, which measures what proportion of a galaxy's total light is emitted within what radius.
A simplified way to represent this is to take the ratio of the radii containing 50% and 90% of the Petrosian flux.

'''

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
import itertools



# copy your splitdata_train_test function here
def splitdata_train_test(data, fraction_training):
    np.random.seed(0)
    np.random.shuffle(data)
    split = int(len(data) * fraction_training)
    return data[:split], data[split:]


# copy your generate_features_targets function here
def generate_features_targets(data):
    # complete the function by calculating the concentrations

    targets = data['class']

    features = np.empty(shape=(len(data), 13))
    features[:, 0] = data['u-g'] + 1
    features[:, 1] = data['g-r']
    features[:, 2] = data['r-i']
    features[:, 3] = data['i-z']
    features[:, 4] = data['ecc']
    features[:, 5] = data['m4_u']
    features[:, 6] = data['m4_g']
    features[:, 7] = data['m4_r']
    features[:, 8] = data['m4_i']
    features[:, 9] = data['m4_z']

    # concentration in u filter
    features[:, 10] = data['petroR50_u'] / data['petroR90_u']
    # concentration in r filter
    features[:, 11] = data['petroR50_r'] / data['petroR90_r']
    # concentration in z filter
    features[:, 12] = data['petroR50_z'] / data['petroR90_z']

    return features, targets


# complete this function by splitting the data set and training a decision tree classifier
def dtc_predict_actual(data):
    # split the data into training and testing sets using a training fraction of 0.7
    train, test = splitdata_train_test(data, 0.7)

    # generate the feature and targets for the training and test sets
    # i.e. train_features, train_targets, test_features, test_targets
    train_features, train_targets = generate_features_targets(train)
    test_features, test_targets = generate_features_targets(test)

    # instantiate a decision tree classifier
    dtc = DecisionTreeClassifier()

    # train the classifier with the train_features and train_targets
    dtc.fit(train_features, train_targets)

    # get predictions for the test_features
    predictions = dtc.predict(test_features)

    # return the predictions and the test_targets
    return predictions, test_targets

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')

'''
if __name__ == '__main__':
    data = np.load('galaxy_catalogue.npy')

    predicted_class, actual_class = dtc_predict_actual(data)

    # Print some of the initial results
    print("Some initial results...\n   predicted,  actual")
    for i in range(10):
        print("{}. {}, {}".format(i, predicted_class[i], actual_class[i]))
'''

# Implement the following function
def calculate_accuracy(predicted_classes, actual_classes):
  return sum(predicted_classes == actual_classes)/len(actual_classes)


if __name__ == "__main__":
  data = np.load('galaxy_catalogue.npy')

  # split the data
  features, targets = generate_features_targets(data)

  # train the model to get predicted and actual classes
  dtc = DecisionTreeClassifier()
  predicted = cross_val_predict(dtc, features, targets, cv=10)

  # calculate the model score using your function
  model_score = calculate_accuracy(predicted, targets)
  print("Our accuracy score:", model_score)

  # calculate the models confusion matrix using sklearns confusion_matrix function
  class_labels = list(set(targets))
  model_cm = confusion_matrix(y_true=targets, y_pred=predicted, labels=class_labels)

  # Plot the confusion matrix using the provided functions.
  plt.figure()
  plot_confusion_matrix(model_cm, classes=class_labels, normalize=False)
  plt.show()
