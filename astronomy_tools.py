"""
One of the most widely used formats for astronomical images is the Flexible Image Transport System.
In a FITS file, the image is stored in a numerical array, which we can load into a NumPy array.

FITS files also have headers which store metadata about the image.

FITS files are a standard format and astronomers have developed many libraries
(in many programming languages) that can read and write FITS files. We're going to use the Astropy module.
"""

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import time
import statistics

hdulist = fits.open('image0.fits')  # (Header/Data Unit) list.
hdulist.info()

data = hdulist[0].data

# Plot the 2D array
plt.imshow(data, cmap=plt.cm.viridis)
plt.xlabel('x-pixels (RA)')
plt.ylabel('y-pixels (Dec)')
plt.colorbar()
plt.show()


# load fits file and finds the brightest pixel
def load_fits(filename):
    hdulist = fits.open(filename)
    data = hdulist[0].data

    arg_max = np.argmax(data)
    max_pos = np.unravel_index(arg_max, data.shape)

    return max_pos


"""
Now we will put everything together and round up this module by calculating the mean of a stack of FITS files. 
Each individual file may or may not have a detected pulsar, but in the final stack you should be able to see a clear detection.
"""


def mean_fits(files):
    n = len(files)
    if n > 0:

        hdulist = fits.open(files[0])
        data = hdulist[0].data
        hdulist.close()  # free up the memory this file has taken up while we were working with it

        for i in range(1, n):
            hdulist = fits.open(files[i])
            data += hdulist[0].data
            hdulist.close()

        mean = data / n
        return mean

"""

Now we're going to look at a different statistical measure — the median, 
which in many cases is considered to be a better measure than the mean due to its robustness to outliers.

However, a naive implementation of the median algorithm can be very inefficient when dealing with large datasets. 

# for 600 000 images we ll stumble upon a memory problem
# must scale the data
# shouldnt hold all the data simultaneously
# Improving would be:
# 1) we can cut our images and make them 50x50 instead of 200x200
# 2) calculate running median


"""

# small reminders on how to test library or algorithm in terms of time and memory usage
# TIME
def time_stat(func, size, ntrials):
  total = 0
  for i in range(ntrials):
    data = np.random.rand(size)
    start = time.perf_counter()
    res = func(data)
    total += time.perf_counter() - start
  return total/ntrials

print('{:.6f}s for statistics.mean'.format(time_stat(statistics.mean, 10**6, 10)))

# SIZE
a = np.zeros(5, dtype=np.int32)
b = np.zeros(5, dtype=np.float64)

for obj in [a, b]:
  print('nbytes         :', obj.nbytes)
  print('size x itemsize:', obj.size*obj.itemsize)


# Return the median image, time of function run and the amount of memory used
# This funtion wont properly work with hundreds thousands images due to memory consumption
# That wasn't the issue for mean (one image at a time) but for median it is
def median_fits(filenames):
    start = time.time()  # Start timer
    # Read in all the FITS files and store in list
    FITS_list = []
    for filename in filenames:
        hdulist = fits.open(filename)
        FITS_list.append(hdulist[0].data)
        hdulist.close()

    # Stack image arrays in 3D array for median calculation
    FITS_stack = np.dstack(FITS_list)

    median = np.median(FITS_stack, axis=2)

    # Calculate the memory consumed by the data
    memory = FITS_stack.nbytes
    # or, equivalently:
    # memory = 200 * 200 * len(filenames) * FITS_stack.itemsize

    # convert to kB:
    memory /= 1024

    stop = time.time() - start  # stop timer
    return median, stop, memory


# Now saving space
# http://www.stat.cmu.edu/~ryantibs/papers/median.pdf
# https://groklearning.com/learn/data-driven-astro/module-2/12/
"""
The full algorithm for a set of  data points works as follows:

Calculate their mean and standard deviation,  and ;
Set the bounds: minval =  and maxval = . Any value >= maxval is ignored;
Set the bin width: width = ;
Make an ignore bin for counting value < minval;
Make  bins for counting values in minval and maxval, e.g. the first bin is minval <= value < minval + width;
Count the number of values that fall into each bin;
Sum these counts until total >= (N + 1)/2. Remember to start from the ignore bin;
Return the midpoint of the bin that exceeded (N + 1)/2.
"""

def running_stats(filenames):
  '''Calculates the running mean and stdev for a list of FITS files using Welford's method.'''
  n = 0
  for filename in filenames:
    hdulist = fits.open(filename)
    data = hdulist[0].data
    if n == 0:
      mean = np.zeros_like(data)
      s = np.zeros_like(data)

    n += 1
    delta = data - mean
    mean += delta/n
    s += delta*(data - mean)
    hdulist.close()

  s /= n - 1
  np.sqrt(s, s)

  if n < 2:
    return mean, None
  else:
    return mean, s

def median_bins_fits(filenames, B):
    # Calculate the mean and standard dev
    mean, std = running_stats(filenames)

    dim = mean.shape  # Dimension of the FITS file arrays

    # Initialise bins
    left_bin = np.zeros(dim)
    bins = np.zeros((dim[0], dim[1], B))
    bin_width = 2 * std / B

    # Loop over all FITS files
    for filename in filenames:
        hdulist = fits.open(filename)
        data = hdulist[0].data

        # Loop over every point in the 2D array
        for i in range(dim[0]):
            for j in range(dim[1]):
                value = data[i, j]
                mean_ = mean[i, j]
                std_ = std[i, j]

                if value < mean_ - std_:
                    left_bin[i, j] += 1

                elif value >= mean_ - std_ and value < mean_ + std_:
                    bin = int((value - (mean_ - std_)) / bin_width[i, j])
                    bins[i, j, bin] += 1

    return mean, std, left_bin, bins


def median_approx_fits(filenames, B):
    mean, std, left_bin, bins = median_bins_fits(filenames, B)

    dim = mean.shape  # Dimension of the FITS file arrays

    # Position of the middle element over all files
    N = len(filenames)
    mid = (N + 1) / 2

    bin_width = 2 * std / B
    # Calculate the approximated median for each array element
    median = np.zeros(dim)
    for i in range(dim[0]):
        for j in range(dim[1]):
            count = left_bin[i, j]
            for b, bincount in enumerate(bins[i, j]):
                count += bincount
                if count >= mid:
                    # Stop when the cumulative count exceeds the midpoint
                    break
            median[i, j] = mean[i, j] - std[i, j] + bin_width[i, j] * (b + 0.5)

    return median