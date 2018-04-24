'''
Using decision trees (tend to overfit the data though)


Using flux magnitudes from the Sloan Digital Sky Survey (SDSS)
catalogue to create colour indices. Flux magnitudes are the total flux (or light) received in five frequency bands (u, g, r, i and z).

The astronomical colour (or colour index) is the difference between the magnitudes of two filters, i.e. u - g or i - z.

This index is one way to characterise the colours of galaxies. For example, if the u-g index is high then the object is
brighter in ultra violet frequencies than it is in visible green frequencies.

Colour indices act as an approximation for the spectrum of the object and are useful for classifying stars into different types.

Calculate the redshift by measuring the flux using a number of different filters and comparing this to models of what we expect galaxies to look like at different redshifts.

Colour indices (u-g, g-i, r-i and i-z) as input and a subset of sources with spectroscopic redshifts as the training dataset.

Decision tree description:
https://groklearning.com/learn/data-driven-astro/module-9/5/


The Sloan data is stored in a NumPy structured array and looks like this:

u	g	r	i	z	...	redshift
19.84	19.53	19.47	19.18	19.11	...	0.54
19.86	18.66	17.84	17.39	17.14	...	0.16
...	...	...	...	...	...	...
18.00	17.81	17.77	17.73	17.73	...	0.47


Control the depth of the decision tree to control overfitting

Use cross validation. Split data into k subsets and train/test on them - k-fold validation
Take the average of the  accuracy measurements to be the overall accuracy of the the model

'''

from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from matplotlib import pyplot as plt


# get_features_targets function here
def get_features_targets(data):
    features = np.zeros((data.shape[0], 4))
    features[:, 0] = data['u'] - data['g']
    features[:, 1] = data['g'] - data['r']
    features[:, 2] = data['r'] - data['i']
    features[:, 3] = data['i'] - data['z']
    targets = data['redshift']
    return features, targets


# median_diff function here
def median_diff(predicted, actual):
    return np.median(np.abs(predicted - actual))


def cross_validate_model(model, features, targets, k):
    kf = KFold(n_splits=k, shuffle=True)

    # initialise a list to collect median_diffs for each iteration of the loop below
    diffs = []

    for train_indices, test_indices in kf.split(features):
        train_features, test_features = features[train_indices], features[test_indices]
        train_targets, test_targets = targets[train_indices], targets[test_indices]

        # fit the model for the current set
        model.fit(train_features, train_targets)

        # predict using the model
        predictions = model.predict(test_features)

        # calculate the median_diff from predicted values and append to results array
        diffs.append(median_diff(predictions, test_targets))

    # return the list with your median difference values
    return diffs


def cross_validate_predictions(model, features, targets, k):
    kf = KFold(n_splits=k, shuffle=True)

    # declare an array for predicted redshifts from each iteration
    all_predictions = np.zeros_like(targets)

    for train_indices, test_indices in kf.split(features):
        # split the data into training and testing
        train_features, test_features = features[train_indices], features[test_indices]
        train_targets, test_targets = targets[train_indices], targets[test_indices]

        # fit the model for the current set
        model.fit(train_features, train_targets)

        # predict using the model
        predictions = model.predict(test_features)

        # put the predicted values in the all_predictions array defined above
        all_predictions[test_indices] = predictions

    # return the predictions
    return all_predictions


if __name__ == "__main__":
    data = np.load('sdss_galaxy_colors.npy')
    features, targets = get_features_targets(data)

    # initialize model with a maximum depth of 19
    dtr = DecisionTreeRegressor(max_depth=19)

    diffs = cross_validate_model(dtr, features, targets, 10)
    print('Differences: {}'.format(', '.join(['{:.3f}'.format(val) for val in diffs])))
    print('Mean difference: {:.3f}'.format(np.mean(diffs)))

    predictions = cross_validate_predictions(dtr, features, targets, 10)

    # calculate and print the rmsd as a sanity check
    diffs = median_diff(predictions, targets)
    print('Median difference: {:.3f}'.format(diffs))

    # plot the results to see how well our model looks
    plt.scatter(targets, predictions, s=0.4)
    plt.xlim((0, targets.max()))
    plt.ylim((0, predictions.max()))
    plt.xlabel('Measured Redshift')
    plt.ylabel('Predicted Redshift')
    plt.show()


    ###

    # Get a colour map
    cmap = plt.get_cmap('YlOrRd')

    # Define our colour indexes u-g and r-i
    u_g = data['u'] - data['g']
    r_i = data['r'] - data['i']

    # Make a redshift array
    redshift = data['redshift']

    # Create the plot with plt.scatter
    plot = plt.scatter(u_g, r_i, s=0.5, lw=0, c=redshift, cmap=cmap)

    cb = plt.colorbar(plot)
    cb.set_label('Redshift')

    # Define your axis labels and plot title
    plt.xlabel('Colour index  u-g')
    plt.ylabel('Colour index  r-i')
    plt.title('Redshift (colour) u-g versus r-i')

    # Set any axis limits
    plt.xlim(-0.5, 2.5)
    plt.ylim(-0.5, 1)

    plt.show()
