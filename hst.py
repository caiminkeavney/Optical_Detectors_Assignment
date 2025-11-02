

from astropy.io import fits
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib import colors
from scipy import ndimage
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
import pandas as pd


def cosmic_remover(directory):
    """
    Reads all FITS files from a given DATA subdirectory, normalizes and combines them 
    into a median-stacked image to remove cosmic rays.

    Parameters:
        directory (str): Name of the subdirectory inside 'DATA' containing FITS files.

    Returns:
        stacked (ndarray): Median-stacked image with cosmic rays removed.
        avg_zeropoint (float): Median photometric zeropoint from all images.
    """

    file = glob(f"DATA/{directory}/*.fits")
    images = []
    headers = {}
    data = {}
    dataheader = {}

    # The data and headers are loaded into dictionary keys.
    for i, fname in enumerate(file, 1):  
        with fits.open(fname) as hdul:
            headers[f"header{i}"] = hdul[0].header
            data[f"data{i}"] = hdul[1].data
            dataheader[f"dataheader{i}"] = hdul[1].header

    # The exposure time for the observation is stored.
    Exp_time = list(headers.values())[0]['EXPTIME']

    # The detector units are converted to spectral flux density.
    zeropoints = []
    for i in range(len(data)):
        header = list(dataheader.values())[i]
        img = list(data.values())[i]

        # The counts per second is calculated.
        if header["BUNIT"] == "COUNTS":
            img = img / Exp_time

        photflam = header["PHOTFLAM"]
        img *= photflam

        # zeropoint is stored for certain image
        zeropoint = header["PHOTZPT"]

        images.append(img)
        zeropoints.append(zeropoint)

    # The stacked image for this filter is generated
    stacked = np.median(images, axis=0)

    # The average zeropoint for the 3 images is found
    avg_zeropoint = float(np.median(zeropoints))
    return stacked, avg_zeropoint


def _sigma_clip(sample, sigma=3.0, niter=10):
    """
    Performs recursive sigma clipping on a dataset.

    Parameters:
        sample (array-like): Input data to clip.
        sigma (float): Clipping threshold in standard deviations.
        niter (int): Maximum number of clipping iterations.

    Returns:
        mask (ndarray): Boolean array marking retained (True) data points.
    """
    # Initial mask with all True pixels is created.
    mask = np.ones_like(sample, dtype=bool)

    # The mask is then iteratively clipped for niter loops.
    for _ in range(niter):

        filtered = sample[mask]

        # If no pixels remain, stop early
        if filtered.size == 0:
            break

        # Compute median and standard deviation of the current valid pixels
        median = np.median(filtered)
        stddev = np.std(filtered)

        if stddev == 0:
            break

        # Create a new mask keeping pixels within a specified range from the median.
        new_mask = np.abs(sample - median) < sigma * stddev

        # If the mask hasn't changed since the last iteration, stop
        if np.all(new_mask == mask):
            break

        # The mask is updated for the next loop.
        mask = new_mask

    # Return final boolean mask (True = pixel kept, False = pixel rejected)
    return mask



def sigma_clip(sample, sigma=3.0, niter=10):
    """
    Computes mean, median, and stddev of sigma-clipped data.

    Parameters:
        sample (ndarray): Input array to clip.
        sigma (float): Clipping threshold in standard deviations.
        niter (int): Maximum number of clipping iterations.

    Returns:
        mean (float): Mean of clipped data.
        median (float): Median of clipped data.
        stddev (float): Standard deviation of clipped data.
    """
    # The recursive clipping function is called to produced a mask of the background.
    mask = _sigma_clip(sample, sigma=sigma, niter=niter)
    clipped = sample[mask]

    # Statistics of the background are returned.
    return np.mean(clipped), np.median(clipped), np.std(clipped)



def background_subtract(image, sigma=3.0, niter=10):
    """
    Estimates and removes background from an image using sigma clipping.

    Parameters:
        image (ndarray): 2D image array.
        sigma (float): Sigma threshold for clipping.
        niter (int): Number of clipping iterations.

    Returns:
        bg_subtracted (ndarray): Background-subtracted image.
        bg_median (float): Estimated background median.
        bg_std (float): Background standard deviation.
        mask (ndarray): Boolean mask of background pixels.
    """
    # The statistics of the background are loaded in using the sigma_clip function
    mean, median, std = sigma_clip(image, sigma=sigma, niter=niter)

    # The background median level is subtracted from the stacked image
    bg_subtracted = image - median

    # A mask of the subtracted mask is kept if needed
    mask = _sigma_clip(image, sigma=sigma, niter=niter)

    return bg_subtracted, median, std, mask


def square_aperture(data, center, box_size):
    """
    Extracts a square cutout around a specified center.

    Parameters:
        data (ndarray): 2D image data.
        center (tuple): (y, x) coordinates of center.
        box_size (int): Size of square box.

    Returns:
        cut (ndarray): Extracted cutout array.
        origin (tuple): (y0, x0) coordinates of cutout origin.
    """
    # coordinates of centre are defined.
    y, x = map(int, center)

    # If box size entered isn't even, it is made so (cannot have half pixel)
    if box_size % 2 == 0:
        box_size += 1


    half = box_size // 2

    ny, nx = data.shape

    # The extents of the square cutout are defined.
    y0 = max(0, y - half)
    y1 = min(ny, y + half + 1)
    x0 = max(0, x - half)
    x1 = min(nx, x + half + 1)

    # The cutout is made
    cut = data[y0:y1, x0:x1]

    return cut, (y0, x0)


def circular_aperture(data, center, radius):
    """
    Extracts a circular aperture around a given center.

    Parameters:
        data (ndarray): 2D image array.
        center (tuple): (y, x) coordinates.
        radius (float): Radius of circular aperture.

    Returns:
        cutout (ndarray): Extracted cutout.
        mask (ndarray): Boolean mask of aperture (True = inside circle).
    """

    box_size = int(np.ceil(2 * radius))

    if box_size % 2 == 0:
        box_size += 1

    # An initial, square cutout is generated.
    cutout, (y0, x0) = square_aperture(data, center, box_size)

    yc, xc = cutout.shape[0] // 2, cutout.shape[1] // 2

    ys, xs = np.indices(cutout.shape)

    # The square aperture is converted into a circular one.
    distance = np.sqrt((ys - yc)**2 + (xs - xc)**2)
    mask = distance <= radius

    return cutout, mask


def gauss_2d(coords, amp, x0, y0, sigma_x, sigma_y):
    """
    2D Gaussian function for curve fitting.

    Parameters:
        coords (tuple): (y, x) coordinates.
        amp (float): Amplitude.
        x0, y0 (float): Centre coordinates.
        sigma_x, sigma_y (float): Standard deviations from the centre.

    Returns:
        gauss (ndarray): Flattened Gaussian array.
    """
    y, x = coords

    # Compute the 2-D gaussian at the point x,y.
    # Creates a 2-D gaussian curve with a centre at x0, y0 and standard deviations sigma_x, sigma_y
    gauss = amp * np.exp(-(((x - x0)**2 / (2 * sigma_x**2)) +
                           ((y - y0)**2 / (2 * sigma_y**2))))
    return gauss.ravel()


def fit_gauss2d(data, p0=None):
    """
    Fits a 2D Gaussian model to image data.

    Parameters:
        data (ndarray): 2D array of image data.
        p0 (list, optional): Initial parameter guesses.

    Returns:
        popt (ndarray): Best-fit parameters.
        pcov (ndarray): Covariance matrix of fit.
    """
    # If no initial parameters are entered, a set is roughly calculated from the flux data.
    if p0 is None:
        a0 = data.max()  # amplitude
        x0 = data.shape[1]/2  # x-peak
        y0 = data.shape[0]/2  # y-peak
        sigx0 = sigy0 = 0.5   # rough estimate of stds.
        p0 = [a0, x0, y0, sigx0, sigy0] 

    y, x = np.indices(data.shape)

    # Gaussian is fit to the data.
    popt, pcov = curve_fit(gauss_2d, (y, x), data.ravel(), p0=p0)
    return popt, pcov


def flux_checker(image, x, y):
    """
    Checks whether the flux at a specific pixel in the image is positive.

    Parameters:
        image (ndarray): 2D array representing the image.
        x (float): X-coordinate (column) of the pixel to check.
        y (float): Y-coordinate (row) of the pixel to check.

    Returns:
        bool: True if the pixel flux is greater than zero, False otherwise.
    """

    ny, nx = image.shape
    xi = int(round(x))
    yi = int(round(y))
    xi = np.clip(xi, 0, nx - 1)
    yi = np.clip(yi, 0, ny - 1)

    flux = image[yi, xi]  
    return flux > 0


def check_ellipticity(sigma_x, sigma_y, etol=0.5):
    """
    Checks if object ellipticity is within tolerance.

    Parameters:
        sigma_x, sigma_y (float): Gaussian widths along axes.
        etol (float): Maximum allowed ellipticity.

    Returns:
        bool: True if ellipticity ≤ etol.
    """
    f = 1 - (min(sigma_x, sigma_y) / max(sigma_x, sigma_y))
    return f <= etol


def star_finder(directory, sigma=5):
    """
    Detects stars in stacked FITS images using sigma thresholding.

    Parameters:
        directory (str): Name of data directory.
        sigma (float): Detection threshold multiplier.

    Returns:
        image_minus_bg (ndarray): Background-subtracted image.
        centroids (list): List of detected star centroids (y, x).
        n_stars (int): Number of detected stars.
        threshold (float): Detection threshold value.
        zeropoint (float): Median zeropoint of images.

    """
    stacked_image, zeropoint = cosmic_remover(directory)

    # Estimate background & threshold
    image_minus_bg, median, std, mask = background_subtract(stacked_image, sigma, niter=10)
    threshold = sigma * std 

    # Apply threshold
    star_mask = (image_minus_bg > threshold)

    # Label & find centroids
    labeled, n_stars = ndimage.label(star_mask)
    centroids = ndimage.center_of_mass(image_minus_bg, labeled, range(1, n_stars + 1))

    return stacked_image, centroids, len(centroids), threshold, zeropoint


def star_checker(x, y, image, radius = 3, annulus_inner=8, annulus_outer=14):

    """
    Checks if a detected source is a valid star using a fixed aperture.

    Parameters:
    x,y (float): Coordinates of candidate star from star_finder
    image (ndarray): stacked image
    radius (int): Radius of circular aperture in pixels
    annulus_inner (int): Inner radius of annulus to be used for local background subtraction
    annulus_outer (int): Outer radius of annulus to be used for local background subtraction
    """
    centre = (y, x)

    cutout, mask = circular_aperture(image, centre, radius)
    if cutout.size == 0:
        return False, radius, None

    # Clip negative pixels
    cutout = np.clip(cutout, 0, None)

    # Background in annulus is calculated
    ann_cutout, _ = circular_aperture(image, centre, annulus_outer)
    yc, xc = ann_cutout.shape[0] // 2, ann_cutout.shape[1] // 2
    ys, xs = np.indices(ann_cutout.shape)
    r = np.sqrt((ys - yc)**2 + (xs - xc)**2)
    ann_mask = (r >= annulus_inner) & (r <= annulus_outer)
    bg_pix = ann_cutout[ann_mask]
    bg_med, bg_std = np.median(bg_pix), np.std(bg_pix)

    # Local background is subtracted from cutout
    star_flux = cutout * mask - bg_med
    peak_flux = star_flux.max()
    # Check if the peak is significantly above nearby background.
    if peak_flux < 2.5 * bg_std:
        return False, radius, bg_med, None

    # Candidate is now fitted with a 2-d Gaussian.
    try:
        popt, _ = fit_gauss2d(star_flux)
        sigmax, sigmay = popt[3], popt[4]

        # ellipticity check
        a = check_ellipticity(sigmax, sigmay, etol=1)

        # width check
        b = (0.3 < sigmax < 8) and (0.3 < sigmay < 8)

        # if either check fails the candidate is rejected.
        test = a and b

    # The Gaussian fits are the most time-intensive part of the script.
    except RuntimeError:
        test = False
        popt = None

    return test, radius, bg_med, popt


def flux_to_magnitude(value, zeropoint=0):
    """
    Converts flux values to magnitudes.

    Parameters:
        value (ndarray): Flux values.
        zeropoint (float): Photometric zeropoint.

    Returns:
        mag (ndarray): Corresponding magnitude.s
    """
    # Apparent Magnitude is zeropoint corrected and returned.
    return -2.5 * np.log10(value) + zeropoint


def get_magnitudes(centroids, image, zeropoint, aperture_radius=3):
    """
    Measures aperture flux and converts to magnitude for each centroid.

    Parameters:
        centroids (list): List of (y, x) coordinates.
        background_subtracted_image (ndarray): Input image.
        aperture_radius (int): Photometric aperture radius.
        zeropoint (float): Photometric zeropoint.

    Returns:
        mags (ndarray): Array of computed magnitudes.
        coords (list): Coordinates used for measurements.
    """

    mags, coords, radii = [], [], []

    for y, x in centroids:
        cut, mask = circular_aperture(image, (y, x), aperture_radius)
        flux = (cut * mask).sum()


        if flux > 0:
            mags.append(flux_to_magnitude(flux, zeropoint))
            coords.append((y, x))
            radii.append(aperture_radius)

    return np.array(mags), coords, radii

def verified_stars(coords, stacked_image, zeropoint):
    """
    This function tests if the star passes the star_checker tests. If it does it is added to the catalogue.

    Parameters:
    coords (list): Set of coordinates to be tested.
    stacked_image (ndarray): Cosmic-Ray removed image
    zeropoint (float): The zeropoint for the filter.

    Returns:
    verified_stars (list): Coordinates of accepted stars
    """
    verified_stars = []
    gaussian_params = []

    _, bg_median, bg_std = sigma_clip(stacked_image)
    median_flux = np.median(stacked_image[stacked_image > bg_median + 3 * bg_std])
    median_flux = max(median_flux, 1e-10)

    for i, (y, x) in enumerate(coords):
        is_star, rad, local_bg, popt = star_checker(
            x, y, stacked_image
        )
        if is_star:
            verified_stars.append((y, x))
            if popt is not None:
                gaussian_params.append((popt[3], popt[4]))

    return verified_stars, gaussian_params

def star_catalogue():
        """
        Places all processed stars into a catalogue, along with their coordinates, and magnitudes in both filters

        Returns:
        Catalogue
        """
        # Green filter candidate stars are found and tested.
        stacked_g, g_stars, n_stars_g, threshold_g, g_zeropoint = star_finder("F555W", 5)
        verified_g, gaussian_g = verified_stars(g_stars, stacked_g, g_zeropoint)

        # Blue filter candidate stars are found and tested.
        stacked_b, b_stars, n_stars_b, threshold_b, b_zeropoint = star_finder("F336W",5)
        verified_b, gaussian_b = verified_stars(b_stars, stacked_b, b_zeropoint)

        # Compute G magnitudes, coords and radii
        g_mags, g_coords, g_radii = get_magnitudes(verified_g, stacked_g,  g_zeropoint , aperture_radius=5)

        # Compute B coordinates only, do not care about B radii
        b_mags, b_coords, _ = get_magnitudes(verified_b, stacked_b, b_zeropoint, aperture_radius=5)



        # List of blue coords made to compare green coords with.
        tree = cKDTree(b_coords)

        matched_catalogue = []
        for i, (g_coord, g_mag, g_rad) in enumerate(zip(g_coords, g_mags, g_radii)):

            # The nearest 1 blue coord to the entered green coord will be paired with it (k=1).
            distance, index = tree.query(g_coord, k=1)

            # The corresponding magnitude for the blue filter is found using the index.
            b_mag = b_mags[index]

            # The green coord is used as the coordinate for the star as an approximation.
            matched_catalogue.append({
                "Star ID": i+1,
                "x": g_coord[1],
                "y": g_coord[0],
                "Aperture Radius": g_rad,
                "G Mag (F555W)": float(g_mag),
                "B Mag (F336W)": float(b_mag),
                "G-B Mag": g_mag - b_mag
            })

        # The catalogue is created using pandas.
        catalogue_df = pd.DataFrame(matched_catalogue) 

        return catalogue_df, gaussian_g



def main():
    # The catalogue of processed stars is loaded in.
    catalogue, gauss_g = star_catalogue()

    # Round numeric columns for readability.
    numeric_cols = ["x", "y", "Aperture Radius", "G Mag (F555W)", "B Mag (F336W)", "G-B Mag"]
    for col in numeric_cols:
        if col in catalogue.columns:
            catalogue[col] = catalogue[col].round(4)

    # Print catalogue
    print(catalogue.to_string(index=False))

    # Plot Hertzsprung-Russell Diagram
    plt.scatter(catalogue["G-B Mag"], catalogue["G Mag (F555W)"], color='k', s=0.4)
    plt.gca().invert_yaxis()
    plt.title("NGC 1261: Hertzsprung–Russell Diagram")
    plt.xlabel("G − B (mag)")
    plt.ylabel("G (mag)")
    plt.tight_layout()
    plt.show()

    # Printing FWHM histogram
    sigma_eff = [((x**2 + y**2)/2)**0.5 for x, y in gauss]

    fwhm_eff = [2.355 * s for s in sigma_eff]

    plt.hist(fwhm_eff, bins=60, color='green', edgecolor='k')
    plt.xlabel("FWHM (pixels)")
    plt.ylabel("Number of Stars")
    plt.title("Histogram of Star FWHM (F555W)")
    plt.show()




if __name__ == "__main__":
    main()
