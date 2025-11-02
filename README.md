# Optical_Detectors_Assignment
The python script found in this repo can be used to produce a star catalogue and H-R Diagram. These will be produced based on the fits images found in DATA. 

## Using the Script
Ensure that the script is in the same directory as the DATA directory, which contains within it the sub-directories for each filter.

The script takes no arguments, so 'python hst.py' should return the results when run in the command line. Ignore warning regarding covariance fitting. The runtime for the script should be in the region of 1 minute.

## Processes

$1.$ First, the spurious cosmic rays are removed from the images by median stacking them for each filter. 

$2.$ The background level median and standard deviation of the stacked image is then determined in order to perform thresholding and find a list of star candidates.

$3.$ The candidates are then put through a star-checking process where their flux, ellipticity and width are checked. A gaussian is fitted to each candidate in order to determine their width.

$4.$ The successful candidates then have their magnitudes calculated in both filters.

$5.$ These magnitudes are used to create a star catalogue as well as a Hertzsprung-Russell Diagram.





