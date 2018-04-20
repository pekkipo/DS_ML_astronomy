'''

When investigating astronomical objects, like active galactic nuclei (AGN),
need to compare data about those objects from different telescopes at different wavelengths.

This requires positional cross-matching to find the closest counterpart within a given radius on the sky.

# we need angular distances because the distance between the objects is not euclidean (not on a flat surface)
# need to use Great Circle Distance

cross-match two catalogues: one from a radio survey,
the AT20G Bright Source Sample (BSS) catalogue and one from an optical survey, the SuperCOSMOS all-sky galaxy catalogue.

Equatoiral coords:
right ascension
inclination


hours-minutes-seconds (HMS) often for right ascension

degrees-minutes-seconds (DMS) traditionally for inclination

To crossmatch two catalogues we need to compare the angular distance between objects on the celestial sphere.

https://groklearning.com/learn/data-driven-astro/module-3/5/

The full catalogue of bright radio sources contains 320 objects.
The catalogue is organised in fixed-width columns, with the format of the columns being:

1: Object catalogue ID number (sometimes with an asterisk)
2-4: Right ascension in HMS notation
5-7: Declination in DMS notation
8-: Other information, including spectral intensities

The SuperCOSMOS all-sky catalogue is a catalogue of galaxies generated from several visible light surveys.
The catalogue uses a comma-separated value (CSV) format. Aside from the first row, which contains column labels, the format is:

1: Right ascension in decimal degrees
2: Declination in decimal degrees
3: Other data, including magnitude and apparent shape
'''

import numpy as np

# Convert to decimal degrees

def hms2dec(h, m, s):
  return 15*(h + m/60 + s/3600)

def dms2dec(d, m, s):
  sign = d/abs(d)
  return sign*(abs(d) + m/60 + s/3600)


def angular_dist(RA1, dec1, RA2, dec2):
    # Convert to radians
    r1 = np.radians(RA1)
    d1 = np.radians(dec1)
    r2 = np.radians(RA2)
    d2 = np.radians(dec2)

    a = np.sin(np.abs(d1 - d2) / 2) ** 2
    b = np.cos(d1) * np.cos(d2) * np.sin(np.abs(r1 - r2) / 2) ** 2

    angle = 2 * np.arcsin(np.sqrt(a + b))

    # Convert back to degrees
    return np.degrees(angle)


def import_bss():
  res = []
  data = np.loadtxt('bss.dat', usecols=range(1, 7))
  for i, row in enumerate(data, 1):
    res.append((i, hms2dec(row[0], row[1], row[2]), dms2dec(row[3], row[4], row[5])))
  return res

def import_super():
  data = np.loadtxt('super.csv', delimiter=',', skiprows=1, usecols=(0, 1))
  res = []
  for i, row in enumerate(data, 1):
    res.append((i, row[0], row[1]))
  return res

"""
Go through each item in the list and pull out its individual RA and declination and use angular_dist 
to find the distance from the given point to the current object.
"""
def find_closest(cat, ra, dec):
    min_dist = np.inf
    min_id = None
    for id1, ra1, dec1 in cat:
        dist = angular_dist(ra1, dec1, ra, dec)
        if dist < min_dist:
            min_id = id1
            min_dist = dist

    return min_id, min_dist

# Naive crossmatch - for large dataset too big time complexity
def crossmatch(cat1, cat2, max_radius):
    matches = []
    no_matches = []
    for id1, ra1, dec1 in cat1:
        closest_dist = np.inf
        closest_id2 = None
        for id2, ra2, dec2 in cat2:
            dist = angular_dist(ra1, dec1, ra2, dec2)
            if dist < closest_dist:
                closest_id2 = id2
                closest_dist = dist

        # Ignore match if it's outside the maximum radius
        if closest_dist > max_radius:
            no_matches.append(id1)
        else:
            matches.append((id1, closest_id2, closest_dist))

    return matches, no_matches

# Optimal solution would be the one implemented in astropy module using kd trees
'''
Astropy will make kd tree out of the second catalogue and will search for a match for each object of the first catalogue
Creating a k-d tree from an astronomy catalogue works like this:

Find the object with the median right ascension, split the catalogue into objects left and right partitions of this
Find the objects with the median declination in each partition, split the partitions into smaller partitions of objects down and up of these
Find the objects with median right ascension in each of the partitions, split the partitions into smaller partitions of objects left and right of these
Repeat 2-3 until each partition only has one object in it
'''

from astropy.coordinates import SkyCoord
from astropy import units as u
from time import time


def crossmatch_tree(coords1, coords2):
    start_time = time()
    max_radius = 5
    matches = []
    no_matches = []

    # Convert to astropy coordinates objects
    coords1_sc = SkyCoord(coords1 * u.degree, frame='icrs')
    coords2_sc = SkyCoord(coords2 * u.degree, frame='icrs')

    # Perform crossmatching
    closest_ids, closest_dists, _ = coords1_sc.match_to_catalog_sky(coords2_sc)

    for id1, (closest_id2, dist) in enumerate(zip(closest_ids, closest_dists)):
        closest_dist = dist.value
        # Ignore match if it's outside the maximum radius
        if closest_dist > max_radius:
            no_matches.append(id1)
        else:
            matches.append([id1, closest_id2, closest_dist])

    time_taken = time() - start_time
    return matches, no_matches, time_taken
