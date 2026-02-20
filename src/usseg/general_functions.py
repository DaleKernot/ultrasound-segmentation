""" A set of functions to segment and extract data from doppler ultrasound scans"""
# Python imports
import traceback
import math
import re
import logging

from rapidfuzz.distance import Levenshtein
# Module imports
import matplotlib.pyplot as plt
from skimage import morphology, measure
import numpy as np
from skimage.measure import find_contours
from skimage.draw import polygon_perimeter
import cv2
from PIL import Image
import scipy
from scipy.ndimage.filters import gaussian_filter
from scipy import signal
from scipy.spatial.distance import cdist
from scipy.signal import find_peaks, peak_widths
import statistics
import scipy.linalg
from sklearn.cluster import DBSCAN
import pandas as pd
import pydicom
import pytesseract
from pytesseract import Output

logger = logging.getLogger(__file__)

root = None  # Assuming you have a reference to the main tkinter window


def execute_on_main_thread_and_wait(func, *args, **kwargs):
    import threading
    import tkinter as tk
    """Executes a function on the main thread and waits for it to complete.

    This function is useful for ensuring that Tkinter objects are manipulated safely from worker threads. Tkinter objects are not thread-safe and can only be manipulated from the main thread.

    Args:
        func: The function to be executed on the main thread.
        *args: The arguments to be passed to the function.
        **kwargs: The keyword arguments to be passed to the function.

    Returns:
        The result of the function, or `None` if the function raised an exception."""

    global root
    if root is None:
        root = tk.Tk()
    if threading.current_thread() == threading.main_thread():
        return func(*args, **kwargs)
    else:
        result = None
        exception = None
        event = threading.Event()

        def callback():
            nonlocal result, exception
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                exception = e
            finally:
                event.set()  # Signal completion

        root.after(0, callback)
        event.wait()  # Block until the function completes on the main thread

        if exception:
            raise exception  # Re-raise any exception that occurred on the main thread

        return result


def extract_dicom_metadata(dicom_file_path):
    """Extract specific metadata from a DICOM file.
    
    Args:
        dicom_file_path (str): Path to the DICOM file.
    
    Returns:
        dict: Dictionary containing extracted DICOM metadata fields.
    """
    try:
        ds = pydicom.dcmread(dicom_file_path, stop_before_pixels=True)
        metadata = {}

        # Step 1: get (0018,6011) Sequence of Ultrasound Regions
        us_region_elem = ds.get((0x0018, 0x6011), None)
        if us_region_elem is None:
            metadata["us_region_sequence_present"] = False
            return metadata

        metadata["us_region_sequence_present"] = True
        metadata["num_us_regions"] = len(us_region_elem.value)
        # Step 2: find PW spectral region (RegionSpatialFormat=3, RegionDataType=3)
        pw_index = None
        for i, item in enumerate(us_region_elem.value):
            rsf_elem = item.get((0x0018, 0x6012), None)  # Region Spatial Format
            rdt_elem = item.get((0x0018, 0x6014), None)  # Region Data Type
            rsf = rsf_elem.value if rsf_elem else None
            rdt = rdt_elem.value if rdt_elem else None

            if rsf == 3 and rdt == 3:
                pw_index = i
                break

        if pw_index is not None:
            pw_region = us_region_elem.value[pw_index]

            def get_tag(tag):
                elem = pw_region.get(tag, None)
                return elem.value if elem else None

            # Bounding box (pixel coords in full image)
            metadata["RegionLocationMinX0"] = get_tag((0x0018, 0x6018))
            metadata["RegionLocationMinY0"] = get_tag((0x0018, 0x601A))
            metadata["RegionLocationMaxX1"] = get_tag((0x0018, 0x601C))
            metadata["RegionLocationMaxY1"] = get_tag((0x0018, 0x601E))

            # Reference pixel (offsets within the region, per your doc)
            metadata["ReferencePixelX0"] = get_tag((0x0018, 0x6020))
            metadata["ReferencePixelY0"] = get_tag((0x0018, 0x6022))

            # Physical units codes
            metadata["PhysicalUnitsXDirection"] = get_tag((0x0018, 0x6024))
            metadata["PhysicalUnitsYDirection"] = get_tag((0x0018, 0x6026))

            # Physical value at the reference pixel
            metadata["ReferencePixelPhysicalValueX"] = get_tag((0x0018, 0x6028))
            metadata["ReferencePixelPhysicalValueY"] = get_tag((0x0018, 0x602A))

            # Physical delta per pixel step
            metadata["PhysicalDeltaX"] = get_tag((0x0018, 0x602C))
            metadata["PhysicalDeltaY"] = get_tag((0x0018, 0x602E))
        else:
            metadata["pw_spectral_region_index"] = None

        metadata["pw_spectral_region_index"] = pw_index

        return metadata

    except Exception as e:
        logger.error(f"Failed to extract DICOM metadata from {dicom_file_path}: {e}")
        return {}

def extract_doppler_from_dicom(dicom_file_path):
    """
    Returns:
        PIL_image (PIL.Image.Image)
        cv2_image (np.ndarray)  # suitable for OpenCV (BGR for colour images)
    """
    ds = pydicom.dcmread(dicom_file_path)  # must read pixels
    arr = ds.pixel_array  # numpy array

    # Handle common cases: RGB (H,W,3) or MONOCHROME (H,W)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        # pydicom gives RGB; PIL expects RGB; OpenCV expects BGR
        PIL_image = Image.fromarray(arr.astype(np.uint8), mode="RGB")
        cv2_image = arr[:, :, ::-1].astype(np.uint8)  # RGB -> BGR
        return PIL_image, cv2_image

    # Grayscale: ensure uint8
    if arr.ndim == 2:
        if arr.dtype != np.uint8:
            # normalize to 0..255 (simple + safe default)
            a = arr.astype(np.float32)
            a -= a.min()
            denom = (a.max() - a.min()) or 1.0
            a = (a / denom * 255.0).astype(np.uint8)
        else:
            a = arr
        PIL_image = Image.fromarray(a, mode="L")
        cv2_image = a  # OpenCV grayscale
        return PIL_image, cv2_image

    raise ValueError(f"Unsupported pixel array shape: {arr.shape}, dtype={arr.dtype}")

def debug_plot_doppler_overlay(PIL_image, metadata, ref_is_relative=True):
    """
    Plot Doppler image with:
      - bounding box corners/outline
      - reference pixel marker

    ref_is_relative=True means ReferencePixelX0/Y0 are offsets from MinX0/MinY0
    (common for USRegionSequence items).
    """
    img = np.array(PIL_image)

    x0 = metadata.get("RegionLocationMinX0")
    y0 = metadata.get("RegionLocationMinY0")
    x1 = metadata.get("RegionLocationMaxX1")
    y1 = metadata.get("RegionLocationMaxY1")
    rx = metadata.get("ReferencePixelX0")
    ry = metadata.get("ReferencePixelY0")

    plt.figure()
    plt.imshow(img)
    plt.axis("off")

    # Bounding box
    if None not in (x0, y0, x1, y1):
        plt.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], linewidth=2)
        plt.scatter([x0, x1, x1, x0], [y0, y0, y1, y1], s=30)

    # Reference pixel
    if None not in (rx, ry):
        if ref_is_relative and None not in (x0, y0):
            rx_abs, ry_abs = x0 + rx, y0 + ry
        else:
            rx_abs, ry_abs = rx, ry
        plt.scatter([rx_abs], [ry_abs], s=70, marker="x")
        plt.text(rx_abs + 5, ry_abs + 5, "ref", fontsize=10)

    plt.show()

def initial_segmentation(input_image_obj):
    """
    Initial segmentation of an ultrasound image.

    The function performs an initial coarse segmentation on an RGB image to identify
    a waveform by converting it to a binary mask, where the waveform is
    represented by white pixels (1) and the background by black pixels (0).
    It applies a thresholding algorithm to differentiate the waveform from
    the background, with additional processing to remove noise, fill holes,
    and adjust the shape of the segmented area. It also calculates the
    bounding box coordinates of the waveform.

    Args:
        input_image_obj (str): Name of file within current directory, or path to a file.

    Returns:
        tuple: tuple containing:
            - **segmentation_mask** (ndarray): A binary array mask showing the coarse segmentation of waveform (1) against background (0).
            - **Xmin** (float): Minimum X coordinate of the segmentation.
            - **Xmax** (float): Maximum X coordinate of the segmentation.
            - **Ymin** (float): Minimum Y coordinate of the segmentation.
            - **Ymax** (float): Maximum Y coordinate of the segmentation.
    """
    # img_RGBA = Image.open(input_image_filename)  # These images are in RGBA form
    img_RGB = input_image_obj
    pixel_data = img_RGB.load()  # Loads a pixel access object, where pixel values can be edited
    # gry = img_RGB.convert("L")  # (returns grayscale version)

    # To threshold the ROI,
    for y in range(img_RGB.size[1]):
        for x in range(img_RGB.size[0]):
            r = pixel_data[x, y][0]  # red component
            g = pixel_data[x, y][1]  # green component
            b = pixel_data[x, y][2]  # blue component
            rgb_values = [r, g, b]
            min_rgb = min(rgb_values)
            max_rgb = max(rgb_values)
            rgb_range = max_rgb - min_rgb  # range across RGB components

            # notice that the spread of values across R, G and B is reasonably small as the colours is a shade of white/grey,
            # It can be isolated by marking pixels with a low range (<50) and resonable brightness (sum or R G B components > 120)
            if (rgb_range < 100 and max_rgb > 120):  # NEEDS REFINING - these values seem to be optimal for the majority tested.
                pixel_data[x, y] = (
                    255,
                    255,
                    255)  # mark all pixels meeting the conditions to white.
            else:
                pixel_data[x, y] = (0, 0, 0)  # If conditions not met, set to black/

            if img_RGB.size[1] > 600:
                if y < 500:  # for some reason x==0 is white, this line negates this.
                    pixel_data[x, y] = (0, 0, 0)
            else:
                if y < 20:
                    pixel_data[x, y] = (0, 0, 0)

    binary_image = np.asarray(img_RGB)  # Make the image an nparray
    pixel_sum = binary_image.sum(-1)  # sum over each pixel (255,255,255)->[765]
    nonzero_pixels = (pixel_sum > 0).astype(bool)  # Change type
    # Some processing to refine the target area
    segmentation_mask = morphology.remove_small_objects(
        nonzero_pixels, max_size=199, connectivity=2
    )  # Remove small objects (noise)
    segmentation_mask = morphology.remove_small_holes(segmentation_mask, max_size=199)  # Fill in any small holes
    segmentation_mask = morphology.erosion(segmentation_mask)  # Erode the remaining binary, this can remove any ticks that may be joined to the main body
    segmentation_mask = morphology.erosion(segmentation_mask)  # Same as above - combine to one line if possible
    segmentation_mask = morphology.dilation(segmentation_mask)  # Dilate to try and recover some of the collateral loss through erosion
    segmentation_mask = segmentation_mask.astype(float)  # Change type

    contours = find_contours(segmentation_mask)  # contours of each object withing segmentation_mask

    xmin_list, xmax_list, ymin_list, ymax_list = (
        [],
        [],
        [],
        [],
    )  # initialise some variables to store max and min values

    for contour in contours:  # find max and mins of each contour
        xmin_list.append(np.min(contour[:, 1]))
        xmax_list.append(np.max(contour[:, 1]))
        ymin_list.append(np.min(contour[:, 0]))
        ymax_list.append(np.max(contour[:, 0]))

    Xmin, Xmax, Ymin, Ymax = (
        np.min(xmin_list),
        np.max(xmax_list),
        np.min(ymin_list),
        np.max(ymax_list),
    )  # find max and min withing the lists.

    return segmentation_mask, Xmin, Xmax, Ymin, Ymax


def define_end_rois(segmentation_mask, Xmin, Xmax, Ymin, Ymax):
    """
    Defines regions of interest (ROIs) to the left and right of a segmented
    waveform for the purpose of searching for axis information. These regions
    are adjacent to the coarse waveform segmentation.

    The function calculates the dimensions of the ROIs based on the given
    segmentation boundaries. These dimensions are then used to identify and
    analyze axis labels and tick marks related to the waveform data.

    Args:
        segmentation_mask (ndarray) : A binary array mask showing the corse segmentation of waveform (1) against background (0).
        Xmin (float) : Minimum X coordinate of the coarse segmentation.
        Xmax (float) : Maximum X coordinate of the coarse segmentation.
        Ymin (float) : Minimum Y coordinate of the coarse segmentation.
        Ymax (float) : Maximum Y coordinate of the coarse segmentation.

    Returns:
        (tuple): tuple containing:
            - **Left_dimensions** (list) : edge points for the left axes ROI [Xmin, Xmax, Ymin, Ymax].
            - **Right_dimensions** (list) : edge points for the left axes ROI [Xmin, Xmax, Ymin, Ymax].
    """

    # For defining the specific ROI either side of the waveform data
    # these ROIs are later used to search for ticks and labels

    Ylim, Xlim = segmentation_mask.shape[0], segmentation_mask.shape[1]  # segmentation_mask is shaped as [y,x] fyi.
    # LHS
    Xmin_L = 0  # Xmin - 50
    Xmax_L = Xmin - 1
    if (Ymin - 125) > 0:
        Ymin_L = Ymin - 75
    else:
        Ymin_L = 1
    Ymax_L = Ylim
    Left_dimensions = [Xmin_L, Xmax_L, Ymin_L, Ymax_L]
    # RHS
    Xmin_R = Xmax
    Xmax_R = Xlim  # Xmax + 50
    if (Ymin - 125) > 0:
        Ymin_R = Ymin - 75
    else:
        Ymin_R = 1

    # Make right ROI extend to the same bottom limit as the left ROI
    Ymax_R = Ylim

    Right_dimensions = [Xmin_R, Xmax_R, Ymin_R, Ymax_R]
    return Left_dimensions, Right_dimensions


def check_inverted_curve(top_curve_mask, Ymax, Ymin, tol=.25):
    """Checks to see if top curve mask is of an inverted waveform

    Args:
        top_curve_mask (ndarray) : A binary array showing a curve along the top of the refined waveform.
        Ymax (float) : Maximum Y coordinate of the segmentation in pixels.
        Ymin (float) : Minimum Y coordinate of the segmentation in pixels.
        tol (float, optional) : If the top curve occupies less than tol * (Ymax - Ymin) rows, then
            the curve is assumed to be inverted (that is True is returned). If the top curve occupies more than
            or equal to this number of rows, the False is returned and the curve is assumed to be non-inverted.
            Defaults to 0.45.

    Returns:
        *return value* (bool) : True if the top curve is of an inverted waveform, False is the top curve is of a non-inverted waveform.
    """
    c_rows = np.where(np.sum(top_curve_mask, axis=1))  # Curve rows
    c_range = np.max(c_rows) - np.min(c_rows)  # Y range of top curve
    return c_range / (Ymax - Ymin) < tol


def segment_refinement(input_image_obj, Xmin, Xmax, Ymin, Ymax, y_zero=None):
    """
    Refines the segmentation of a waveform within specified bounds, improving 
    the separation between the waveform and background. It processes a given 
    image object within the region of interest (ROI) and generates masks 
    for the waveform and its top curve.

    Args:
        input_image_obj (ndarray): An image object, typically read from a file
            using a library such as OpenCV.
        Xmin (float): Minimum X coordinate of the segmentation in pixels, 
            defining the left boundary of the ROI.
        Xmax (float): Maximum X coordinate of the segmentation in pixels,
            defining the right boundary of the ROI.
        Ymin (float): Minimum Y coordinate of the segmentation in pixels,
            defining the bottom boundary of the ROI.
        Ymax (float): Maximum Y coordinate of the segmentation in pixels,
            defining the top boundary of the ROI.
        y_zero (float, optional): Estimated y-coordinate of the physical 0-line
            in image pixels. Currently unused, but accepted for future use.

    Returns:
        (tuple) : tuple containing:
            - **refined_segmentation_mask** (ndarray): A binary array mask showing the refined segmentation of the waveform (value 1) against the background (value 0).
            - **top_curve_mask** (ndarray): A binary array representing a curve along the top of the refined waveform segmentation.
            - **top_curve_coords** (ndarray): An array of coordinates (row, column) for the top curve of the waveform.
    """

    # 1) Produce the refined binary segmentation mask
    refined_segmentation_mask = refine_waveform_segmentation(
        input_image_obj, Xmin, Xmax, Ymin, Ymax
    )

    # 2) From that mask, derive the top-curve representation
    top_curve_mask, top_curve_coords = compute_top_curve(
        refined_segmentation_mask, Ymin, Ymax, y_zero=y_zero
    )

    return refined_segmentation_mask, top_curve_mask, top_curve_coords


def refine_waveform_segmentation(input_image_obj, Xmin, Xmax, Ymin, Ymax):
    """
    Core segmentation routine used by ``segment_refinement``.

    This function performs the image thresholding and all morphological
    operations required to obtain the final refined binary mask of the
    waveform region. It does **not** compute the top-curve.
    """

    # Refine segmentation to increase smoothing
    # Save output to .txt file to load later.

    image = input_image_obj
    input_image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresholded_image = cv2.threshold(input_image_gray, 30, 255, 0)
    thresholded_image[:, int(Xmax): -1] = 0
    thresholded_image[:, 0: int(Xmin) - 1] = 0
    thresholded_image[0: int(Ymin) - 50, :] = 0
    thresholded_image[int(Ymax): -1, :] = 0
    main_ROI = thresholded_image  # Main ROI

    binary_image = main_ROI  # Make the image an nparray
    nonzero_pixels = (binary_image > 0).astype(bool)  # Change type
    # Some processing to refine the target area
    refined_segmentation_mask = morphology.remove_small_objects(
        nonzero_pixels, max_size=199, connectivity=2
    )  # Remove small objects (noise)
    refined_segmentation_mask = morphology.remove_small_holes(refined_segmentation_mask, max_size=199)  # Fill in any small holes
    refined_segmentation_mask = morphology.erosion(refined_segmentation_mask)  # Erode the remaining binary, this can remove any ticks that may be joined to the main body
    refined_segmentation_mask = morphology.erosion(refined_segmentation_mask)  # Same as above - combine to one line if possible
    refined_segmentation_mask = morphology.dilation(refined_segmentation_mask)  # Dilate to try and recover some of the collateral loss through erosion

    refined_segmentation_mask = morphology.dilation(refined_segmentation_mask)
    refined_segmentation_mask = morphology.remove_small_holes(refined_segmentation_mask, max_size=999)
    refined_segmentation_mask = morphology.closing(refined_segmentation_mask)
    refined_segmentation_mask = refined_segmentation_mask.astype(int)
    refined_segmentation_mask = scipy.signal.medfilt(refined_segmentation_mask, 3)

    # assuming mask is a binary image
    # label and calculate parameters for every cluster in mask
    labelled = measure.label(refined_segmentation_mask)
    rp = measure.regionprops(labelled)

    # get size of largest cluster
    sizes = sorted([i.area for i in rp])
    refined_segmentation_mask = refined_segmentation_mask.astype(bool)
    # remove everything smaller than second-largest area + 10
    try:
        threshold = sizes[-2] + 10
        refined_segmentation_mask = morphology.remove_small_objects(
            refined_segmentation_mask, max_size=threshold - 1
        )
    except Exception:
        pass
    refined_segmentation_mask = refined_segmentation_mask.astype(float)
    # refined_segmentation_mask[rr, cc] = 1 #set color white

    blurred = gaussian_filter(refined_segmentation_mask, sigma=7)
    refined_segmentation_mask = (blurred > 0.5) * 1

    return refined_segmentation_mask


def compute_top_curve(refined_segmentation_mask, Ymin, Ymax, y_zero=None):
    """
    Given a refined segmentation mask, compute the top-curve representation.

    This function reads the supplied mask (without mutating it) and derives:
      - ``top_curve_mask``: a thin mask along the top of the waveform.
      - ``top_curve_coords``: the (row, column) coordinates of that curve.
    """
    # Work on a copy to avoid mutating the original refined_segmentation_mask
    mask = refined_segmentation_mask.copy()

    # label and calculate parameters for every cluster in mask
    labelled = measure.label(mask)
    rp = measure.regionprops(labelled)

    ws = morphology.erosion(mask).astype(float)
    top_curve_mask = mask - ws

    keep = "upper"

    if y_zero is not None:
        # Use the physical baseline (y=0) instead of centroid to decide which
        # side of the waveform to keep.
        y0 = float(y_zero)
        # Find all rows that contain part of the curve
        c_rows = np.where(np.sum(top_curve_mask, axis=1))[0]
        above = np.sum(c_rows < y0)
        below = np.sum(c_rows > y0)

        inverted = below > above  # majority of curve lies below baseline

        if not inverted:
            # Keep rows at or above the baseline (waveform above baseline)
            for x in range(int(np.floor(y0)) + 1, top_curve_mask.shape[0]):
                top_curve_mask[x, :] = 0
            keep = "upper"
        else:
            # Keep rows at or below the baseline (waveform below baseline)
            for x in range(0, int(np.ceil(y0))):
                top_curve_mask[x, :] = 0
            keep = "lower"
    else:
        # Fallback: original centroid-based logic
        for x in range(int(rp[0].centroid[0]), top_curve_mask.shape[0]):
            top_curve_mask[x, :] = 0

        # If inverted, keep only below centroid -> "bottom half"
        if check_inverted_curve(top_curve_mask, Ymax, Ymin):
            top_curve_mask = mask - ws
            for x in range(0, int(rp[0].centroid[0])):
                top_curve_mask[x, :] = 0
            keep = "lower"  # inverted means you want the bottom boundary

    # --- Ray trace / first-hit per column (removes double lines) ---
    top_curve_mask = keep_one_pixel_per_column(top_curve_mask, keep=keep)

    top_curve_coords = np.column_stack(np.nonzero(top_curve_mask))

    return top_curve_mask.astype(int), top_curve_coords


def keep_one_pixel_per_column(mask: np.ndarray, keep: str = "upper") -> np.ndarray:
    """
    Keep at most one True pixel per column in a binary mask.

    keep="upper": visually uppermost pixel (smallest row index).
    keep="lower": visually lowermost pixel (largest row index).
    """
    m = (mask > 0)
    out = np.zeros_like(m, dtype=bool)

    rows, cols = np.nonzero(m)
    if cols.size == 0:
        return out

    for c in np.unique(cols):
        r = rows[cols == c]
        out[(r.min() if keep == "upper" else r.max()), c] = True

    return out


def search_for_ticks(input_image_obj, side, left_dimensions, right_dimensions):
    """
    Search for tick marks on either the left or right axis of an image.

    This function locates contours that resemble tick marks by processing an
    image. It converts the image to grayscale, applies thresholding to generate
    a binary image, and identifies contours. The contours are then filtered
    based on their geometric properties. It returns the contours, the region of
    interest (ROI) of the axis, the center points of the ticks, and additional
    details of the processing.

    Args:
        input_image_obj (str) : Name of file within current directory, or path to a file.
        side (str) : Indicates the 'Left' or 'Right' axes.
        left_dimensions (list) : edge points for the left axes ROI [Xmin, Xmax, Ymin, Ymax].
        right_dimensions (list) : edge points for the left axes ROI [Xmin, Xmax, Ymin, Ymax].

    Returns:
        (tuple) : tuple containing:
            - **Cs** (tuple) : list of contours for ticks found.
            - **ROIAX** (ndarray) : narray defining the ROI to search for the axes.
            - **CenPoints** (list) : center points for the ticks identified.
            - **onY** (list) : indexes of the contours which lie on the target x plane.
            - **BCs** (list) : Contours of the objects which lie on the target x plane.
            - **TYLshift** (intc) : shift in the x coordninate bounds - reducing the axes ROI in which to search for axes text.
            - **thresholded_image** (ndarray) : Threshold values iterated through.
            - **Side** (str) : Indicates the 'Left' or 'Right' axes.
            - **Left_dimensions** (list) : edge points for the left axes ROI [Xmin, Xmax, Ymin, Ymax].
            - **Right_dimensions** (list) : edge points for the left axes ROI [Xmin, Xmax, Ymin, Ymax].
            - **ROI2** (ndarray) : Secondary ROI, stores contour detection data during tick search.
            - **ROI3** (ndarray) : Axes ROI used for visualisation (not used - only initialised here).
    """

    image = input_image_obj
    thresholded_image = image

    if side == "Left":
        ROIAX = thresholded_image[
                int(left_dimensions[2]): int(left_dimensions[3]),
                int(left_dimensions[0]): int(left_dimensions[1]),
                ]  # Right ROI
    elif side == "Right":
        ROIAX = thresholded_image[
                int(right_dimensions[2]): int(right_dimensions[3]),
                int(right_dimensions[0]): int(right_dimensions[1]),
                ]  # Left ROI

    RGBnp = np.array(ROIAX)  # convert images to array (not sure needed)
    RGBnp[RGBnp <= 10] = 0  # Make binary with low threshold
    RGBnp[RGBnp > 10] = 1
    BinaryNP = RGBnp[:, :, 0]  # [0,0,0]->[0]

    binary_image = BinaryNP
    pixel_sum = binary_image  # .sum(-1)  # sum over color (last) axis
    nonzero_pixels = (pixel_sum > 0).astype(bool)
    W = morphology.remove_small_objects(nonzero_pixels, max_size=15, connectivity=2)
    W = W.astype(float)

    im = input_image_obj
    input_image_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresholded_image = cv2.threshold(input_image_gray, 127, 255, 0)

    if side == "Left":
        ROIAX = thresholded_image[
                int(left_dimensions[2]): int(left_dimensions[3]),
                int(left_dimensions[0]): int(left_dimensions[1]),
                ]  # Right ROI
        TGT = 48
    elif side == "Right":
        TGT = 2
        ROIAX = thresholded_image[
                int(right_dimensions[2]): int(right_dimensions[3]),
                int(right_dimensions[0]): int(right_dimensions[1]),
                ]  # Left ROI

    ROI2 = np.zeros(np.shape(ROIAX))
    ROI3 = np.zeros(np.shape(ROIAX))
    # plt.imshow(ROI)
    # plt.show()

    contours, hierarchy = cv2.findContours(
        ROIAX, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(ROI2, contours, -1, [255], 1)
    Cs = list(contours)  # list the contour coordinates as array
    if side == "Right":
        # Object-based approach: identify tick objects and remove columns with too many overlapping ticks
        # Recompute W from the final ROIAX for consistency
        W_right = (ROIAX > 0).astype(bool)
        W_right = morphology.remove_small_objects(W_right, max_size=15, connectivity=2)
        
        # Label connected components
        labels = measure.label(W_right, connectivity=2)
        props = measure.regionprops(labels)
        
        # Define size thresholds for tick-like objects
        min_tick_area = 10  # Minimum area for a tick
        max_tick_area = 200  # Maximum area for a tick (adjust based on your images)
        max_tick_height = 5  # Maximum height for a tick
        
        tick_objects = []  # Indices of tick-like components
        
        for idx, prop in enumerate(props, start=1):
            area = prop.area
            bbox = prop.bbox  # (min_row, min_col, max_row, max_col)
            height = bbox[2] - bbox[0]
            
            # Classify as tick if area and height are within tick-like ranges
            if min_tick_area <= area <= max_tick_area and height < max_tick_height:
                tick_objects.append(idx)
        
        # Visualize tick objects (for debugging - comment out when not needed)
        if tick_objects:
            tick_mask = np.zeros_like(W_right, dtype=bool)
            for tick_idx in tick_objects:
                tick_mask[labels == tick_idx] = True
            
            # Create RGB visualization: red for tick objects, gray for everything else
            vis_image = np.zeros((*W_right.shape, 3), dtype=np.uint8)
            vis_image[W_right] = [128, 128, 128]  # Gray for all objects
            vis_image[tick_mask] = [255, 0, 0]  # Red for tick objects
            
            #plt.figure(figsize=(12, 8))
            #plt.imshow(vis_image)
            #plt.title(f'Right axis ROI: Tick objects (red) vs all objects (gray)\nFound {len(tick_objects)} tick objects')
            #plt.axis('off')
            #plt.show()
        
        # For each column, count how many tick objects touch it
        roi_height, roi_width = ROI2.shape
        columns_to_remove = []
        
        for Column in range(roi_width):
            # Count how many distinct tick objects touch this column
            tick_count = 0
            for tick_idx in tick_objects:
                if np.any(labels[:, Column] == tick_idx):
                    tick_count += 1
            
            # Remove column if too many tick objects overlap (likely axis-dominated area)
            # or if no tick objects touch it (pure axis/noise)
            if tick_count > 20 or tick_count == 0:
                columns_to_remove.append(Column)
        
        # Remove identified columns from ROI2
        for Column in columns_to_remove:
            ROI2[:, Column] = 0

    ROI2 = ROI2.astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        ROI2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    cv2.drawContours(ROI2, contours, -1, [255], 1)
    Cs = list(contours)  # list the contour coordinates as array
    CsX = []  # X coord of the center of each contour
    CsY = []  # X coord of the center of each contour
    for C in Cs:
        # compute the center of the contour
        cx = 0
        cy = 0
        for rgb_values in C:
            cx += rgb_values[0][0]
            cy += rgb_values[0][1]
        CsX.append(int(cx / len(C)))
        CsY.append(int(cy / len(C)))

    I = len(Cs)  # number of contours
    lengths = []  # Initialise variable to fill with contour lengths

    for i in range(0, I):
        lengths.append(Cs[i].size)  # fill with contour lengths

    lengths = np.array(lengths)  # make an array
    ids = np.where(lengths > 0)  # indexes of the lengths greater than 0
    ids = ids[0]  # because ids looks like (array([...])) and we want array([...])
    ids = ids[
          ::-1
          ]  # reverse order so indexes run fron high to low, this is needed for the next loop
    onY, BCs, Xs, EndPoints, CenPoints = (
        [],
        [],
        [],
        [],
        [],
    )  # Initialise some variables

    all = []
    for TGT in range(0, int(right_dimensions[1]) - int(right_dimensions[0])):
        count = 0
        Ctest = 0
        for id in ids:
            Ctest = np.reshape(Cs[id], (-1, 2))
            x_values = [i[0] for i in Ctest]
            if TGT in x_values:
                count = count + 1

        all.append(count)

    peaks, vals = signal.find_peaks(all, height=3)  # Miss last 20 pixels as
    if side == "Left":
        maxID = np.argmax(peaks)
    elif side == "Right":
        peak_wid = signal.peak_widths(all, peaks)
        maxID = np.argmax(vals["peak_heights"] * peak_wid[0])

    TGT = peaks[maxID]
    # TGT = all.index(max(all)) # The target is the X coord that most object lie on.

    for id in ids:
        Ctest = np.reshape(Cs[id], (-1, 2))
        x_values = [i[0] for i in Ctest]
        if (
                TGT in x_values
        ):  # Looks if any contour coords are on a line (2,:), this is close to ROI bounds but not in contact.
            tempXs, tempYs = (
                [],
                [],
            )  # Clear/Initialise temporary X Y coordinate stores
            # cv2.drawContours(ROI3, Cs[id], -1,[255], 1)
            onY.append(id)  # record the contour index that meets criteria
            BCs.append(Cs[id])  # ?
            for l in range(0, len(Cs[id])):
                Xs.append(Cs[id][l][0][0])  # Needed?
                tempXs.append(Cs[id][l][0][0])  # Store X coords
                tempYs.append(Cs[id][l][0][1])  # Store Y coords

            MAXX = max(tempXs)  # Max X from this contour
            MINX = min(tempXs)  # Min X from this contour
            MAXY = max(tempYs)  # Max Y from this contour
            MINY = min(tempYs)  # Min Y from this contour
            if side == "Left":
                index = tempXs.index(
                    MINX
                )  # Index of the max X - this is the "end point" of Side == "Right"
                EndPoints.append(
                    tempXs[index]
                )  # Save end point (Might be redundant with new method?)
            elif side == "Right":
                index = tempXs.index(
                    MAXX
                )  # Index of the max X - this is the "end point" of Side == "Right"
                EndPoints.append(
                    tempXs[index]
                )  # Save end point (Might be redundant with new method?)
            CenPoints.append(
                [int((MAXX + MINX) / 2), int((MAXY + MINY) / 2)]
            )  # Calc center point as (0.5*(MaxX+MinX),0.5*(MaxY+MinY))

    def reject_outliers(data, m=8.0):
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d / (mdev if mdev else 1.0)
        outdata = []
        badIDS = []
        for i in range(0, len(data)):
            if s[i] < m:
                outdata.append(data[i])
            else:
                badIDS.append(i)

        return outdata, badIDS

    if side == "Right":
        TYLshift = max(
            EndPoints
        )  # The shift reduces the ROIAX to avoid intaining the ticks, as these can be confused as '-' symbols
    elif side == "Left":
        EndPoints, badIDS = reject_outliers(EndPoints)
        try:
            # cv2.drawContours(ROI3, Cs[badIDS[0]], -1,[255], 1)
            CenPoints.pop(badIDS[0])
            Cs.pop(badIDS[0])
            onY.pop(badIDS[0])
            BCs.pop(badIDS[0])
        except Exception:
            pass

        TYLshift = min(
            EndPoints
        )  # The shift reduces the ROIAX to avoid intaining the ticks, as these can be confused as '-' symbols
    Cs = tuple(Cs)  # Change to tuple?

    if side == "Left":
        ROIAX = thresholded_image[
                int(left_dimensions[2]): int(left_dimensions[3]),
                int(left_dimensions[0]): int(left_dimensions[0] + TYLshift),
                ]  # Right ROI
    elif side == "Right":
        ROIAX = thresholded_image[
                int(right_dimensions[2]): int(right_dimensions[3]),
                int(right_dimensions[0] + TYLshift): int(right_dimensions[1]),
                ]  # Left ROI

    return (
        Cs,
        ROIAX,
        CenPoints,
        onY,
        BCs,
        TYLshift,
        thresholded_image,
        side,
        left_dimensions,
        right_dimensions,
        ROI2,
        ROI3,
    )


def search_for_labels(
        Cs,
        ROIAX,
        CenPoints,
        onY,
        BCs,
        TYLshift,
        Side,
        Left_dimensions,
        Right_dimensions,
        input_image_obj,
        ROI2,
        ROI3,
):
    """
    Searches for labels within specified regions of an image, extracts text,
    and attempts to associate it with the nearest tick marks.

    This function iterates over a range of threshold values to optimize text
    extraction from an image. It uses OpenCV for image processing and Pytesseract
    for OCR to extract text. The text is then attempted to be matched to the
    nearest tick marks based on center points. The function adapts to the side of
    the image being analyzed (left or right) and draws rectangles around the
    detected text. It also warns if characters are too close.

    Args:
        Cs (tuple): List of center points.
        ROIAX (ndarray): Region of Interest (ROI) array for the X axis, modified
        within the function.
        CenPoints (list): List of center points for detected objects/ticks.
        onY (list): 
        BCs (list): List of baseline coordinates, likely for the tick marks.
        TYLshift (int): Shift along the Y axis for thresholding, specific to the 
        Left side.
        Side (str): Side of the image being processed ('Left' or 'Right').
        Left_dimensions (list): Dimensions for the left ROI.
        Right_dimensions (list): Dimensions for the right ROI.
        input_image_obj (ndarray): Image object to be processed.
        ROI2 (ndarray): Secondary ROI, stores contour detection data during tick search.
        ROI3 (ndarray): Axes ROI used for visualisation.

    Returns:
        (tuple): tuple containing:
            - **ROIAX** (ndarray): narray defining the ROI to search for the axes.
            - **number** (list): A list of label values found on axis.
            - **positions** (list): A list of positions of the label values.
            - **empty_to_fill** (ndarray): A array showing bounding boxes on image.
    """
    extracted_text_data = None
    for thresh_value in np.arange(100, 190, 5):  # Threshold to optimise the resulting text extraction.
        image = input_image_obj
        input_image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresholded_image = cv2.threshold(input_image_gray, thresh_value, 255, 0)
        if Side == "Left":
            ROIAX = thresholded_image[
                    int(Left_dimensions[2]): int(Left_dimensions[3]),
                    int(Left_dimensions[0]): int(Left_dimensions[0] + TYLshift),
                    ]  # Right ROI
        elif Side == "Right":
            ROIAX = thresholded_image[
                    int(Right_dimensions[2]): int(Right_dimensions[3]),
                    int(Right_dimensions[0] + TYLshift): int(Right_dimensions[1]),
                    ]  # Left ROI

        extracted_text_data = pytesseract.image_to_data(
            ROIAX,
            output_type=Output.DICT,
            config="--psm 11 -c tessedit_char_whitelist=-0123456789",
        )
        number = []
        for i in range(len(extracted_text_data["text"])):
            if extracted_text_data["text"][i] != "":
                number.append(extracted_text_data["text"][i])

        retry = 0
        for num in number:
            try:
                if (float(num) / 5).is_integer() == 0:
                    retry += 1
            except Exception:
                pass

        if retry == 0:
            break

    # d = pytesseract.image_to_data(
    #     ROIAX,
    #     output_type=Output.DICT,
    #     config="--psm 11 -c tessedit_char_whitelist=-0123456789",
    # )
    n_boxes = len(extracted_text_data["level"])
    CenBox = []  # Initialise variable to populate with box center coords
    for i in range(1, n_boxes):  # dont start from 0 as the first index is redundant
        if Side == "Left":
            (x, y, wi, h) = (
                extracted_text_data["left"][i],
                extracted_text_data["top"][i],
                extracted_text_data["width"][i],
                extracted_text_data["height"][i],
            )  # define (Xleft, Ytop, width, height) of each object from the dictionary
        elif Side == "Right":
            (x, y, wi, h) = (
                extracted_text_data["left"][i] + TYLshift,
                extracted_text_data["top"][i],
                extracted_text_data["width"][i],
                extracted_text_data["height"][i],
            )  # define (Xleft, Ytop, width, height) of each object from the dictionary

        o = i / 4  # we get 4 repeats for each real box, so this reduces that to 1.
        if o.is_integer():
            if Side == "Left":
                CenBox.append(
                    [
                        (extracted_text_data["left"][i] + (extracted_text_data["width"][i] / 2)),
                        (extracted_text_data["top"][i] + (extracted_text_data["height"][i] / 2)),
                    ]
                )  # calculate the center point of each bounding box
            elif Side == "Right":
                CenBox.append(
                    [
                        (extracted_text_data["left"][i] + TYLshift + (extracted_text_data["width"][i] / 2)),
                        (extracted_text_data["top"][i] + (extracted_text_data["height"][i] / 2)),
                    ]
                )  # calculate the center point of each bounding box
        cv2.rectangle(
            ROI3, (x, y), (x + wi, y + h), (255), 2
        )  # Draw the rectangles on the ROI

    for i in range(0, len(CenBox)):
        dists = cdist([CenBox[i]], CenBox)
        dists[0][i] = dists.max()
        if dists.min() < 20:  # detect if characters are too close - but what if they are?
            logger.warning("Characters too close")

    try:
        # Find the nearest tick for each number
        dist = []  # Initlise distance variable
        Mindex = np.zeros(len(CenBox))  # Initlialise Min index variable
        for txt in range(0, len(CenBox)):  # For all axes label text boxes
            for tck in range(0, len(CenPoints)):  # For all axes ticks
                dist.append(
                    math.sqrt(
                        (CenBox[txt][0] - CenPoints[tck][0]) ** 2 + (CenBox[txt][1] - CenPoints[tck][1]) ** 2
                    )
                )  # Distance between current text box and all ticks
            MIN = min(dist)  # Find the shortest distance to a tick
            Mindex[txt] = dist.index(
                MIN
            )  # Identify the index of the tick at the nearest distance and store.
            dist = []  # Clear dist variable

        positions = []  # Initialise position variable
        for id in Mindex:  # for each index in the Min index store
            # Store the closest centerpoints as found in the previous loop
            #  add ROI components to make positions relative to overall image.
            if Side == "Left":  # Adjust for the left side
                positions.append(
                    [
                        CenPoints[int(id)][0] + int(Left_dimensions[0]),
                        CenPoints[int(id)][1] + int(Left_dimensions[2]),
                    ]
                )
            elif Side == "Right":  # Or adjust for the right side
                positions.append(
                    [
                        CenPoints[int(id)][0] + int(Right_dimensions[0]),
                        CenPoints[int(id)][1] + int(Right_dimensions[2]),
                    ]
                )

        Mindex = Mindex[::-1]  # Reverse order so runs from high to low

        # for id in Mindex: # Common sense check - are all ticks evenly space?
        #     Failed_Indexes.append(BCs[int(id)]) # Order ticks lowest to highest?

        Final_TickIDS = []  # Init variable to store final tick indexs
        Final_CenPoints = []
        IDSL = []
        Mindex = sorted(Mindex, reverse=True)
        for id in Mindex:
            Final_TickIDS.append(BCs[int(id)])  # Order ticks lowest to highest?
            IDSL.append(int(id))
            Final_CenPoints.append(CenPoints[int(id)])

    except Exception:  # if this step fails, a backup is to assume center of text box is the tick
        extracted_text_data = pytesseract.image_to_data(
            ROIAX,
            output_type=Output.DICT,
            config="--psm 11 -c tessedit_char_whitelist=-0123456789",
        )
        n_boxes = len(extracted_text_data["level"])
        CenBox = []  # Initialise variable to populate with box center coords
        for i in range(1, n_boxes):  # dont start from 0 as the first index is redundant
            if Side == "Left":
                (x, y, wi, h) = (
                    extracted_text_data["left"][i],
                    extracted_text_data["top"][i],
                    extracted_text_data["width"][i],
                    extracted_text_data["height"][i],
                )  # define (Xleft, Ytop, width, height) of each object from the dictionary
            elif Side == "Right":
                (x, y, wi, h) = (
                    extracted_text_data["left"][i] + TYLshift,
                    extracted_text_data["top"][i],
                    extracted_text_data["width"][i],
                    extracted_text_data["height"][i],
                )  # define (Xleft, Ytop, width, height) of each object from the dictionary

            o = i / 4  # we get 4 repeats for each real box, so this reduces that to 1.
            if o.is_integer():
                if Side == "Left":
                    CenBox.append(
                        [
                            (extracted_text_data["left"][i] + (extracted_text_data["width"][i] / 2)),
                            (extracted_text_data["top"][i] + (extracted_text_data["height"][i] / 2)),
                        ]
                    )  # calculate the center point of each bounding box
                elif Side == "Right":
                    CenBox.append(
                        [
                            (extracted_text_data["left"][i] + TYLshift + (extracted_text_data["width"][i] / 2)),
                            (extracted_text_data["top"][i] + (extracted_text_data["height"][i] / 2)),
                        ]
                    )  # calculate the center point of each bounding box
            cv2.rectangle(
                ROI3, (x, y), (x + wi, y + h), (255), 2
            )  # Draw the rectangles on the ROI

        for i in range(0, len(CenBox)):
            dists = cdist([CenBox[i]], CenBox)
            dists[0][i] = dists.max()
            if dists.min() < 20:
                logger.warning("Characters too close")

        Final_CenPoints = CenBox

    # Failed_Indexes = []

    dists = cdist(Final_CenPoints, Final_CenPoints)

    def index_list(dists_inner):
        lst = list(dists_inner)
        length = len(lst)
        dist_inner = []
        for i in lst:
            if lst.index(0) > lst.index(i):
                diff = lst.index(0) - lst.index(i)
                dist_inner.append(diff)
            elif lst.index(0) < lst.index(i):
                diff = abs(lst.index(i)) - abs(lst.index(0))
                dist_inner.append(diff)
            elif lst.index(0) == lst.index(i):
                dist_inner.append(0)

        dist_divided = []
        for i in range(0, length):
            dist_divided.append(lst[i] / dist_inner[i])

        return dist_divided

    # ordered_indexs1 = index_list(dists[0])
    # ordered_indexs2 = index_list(dists[1])
    # ordered_indexs3 = index_list(dists[2])
    # ordered_indexs4 = index_list(dists[3])
    # ordered_indexs5 = index_list(dists[4])
    # ordered_indexs6 = index_list(dists[5])

    cv2.drawContours(ROI3, Final_TickIDS, -1, [255], 1)

    def correct_number_format(s):
        # Check numbers don't end in a '-'
        # Using regex to identify the pattern
        pattern = r'(-?\d+)-$'
        return re.sub(pattern, r'\1', s)

    number = []
    for i in range(len(extracted_text_data["text"])):
        if extracted_text_data["text"][i] != "" and extracted_text_data["text"][i] != "-":
            number.append(correct_number_format(extracted_text_data["text"][i]))

    empty_to_fill = np.zeros((image.shape[0], image.shape[1]))

    if Side == "Left":
        empty_to_fill[
          int(Left_dimensions[2]): int(Left_dimensions[3]),
          int(Left_dimensions[0]): int(Left_dimensions[1]),
        ] = ROI3  # Right ROI
    elif Side == "Right":
        empty_to_fill[
          int(Right_dimensions[2]): int(Right_dimensions[3]),
          int(Right_dimensions[0]): int(Right_dimensions[1]),
        ] = ROI3  # Left ROI

    return ROIAX, number, positions, empty_to_fill


def validate_axis_ticks(numbers, positions, side, spacing_tol=0.2):
    """
    Validate and (lightly) correct axis tick labels and their pixel coordinates.

    - Requires at least two ticks, and matching lengths for values/positions.
    - Ensures that, when ordered along the axis (by y pixel), numeric values are
      monotonic (all increasing or all decreasing). If this fails, the inputs
      are returned unchanged.
    - Optionally checks that successive slopes (Δvalue / Δy) are roughly
      consistent. If *exactly two consecutive* slopes are strong outliers, the
      tick between them is treated as suspect and dropped from the returned
      lists. More complex patterns only generate warnings.

    The original inputs are never mutated. A cleaned copy is returned.

    Args:
        numbers (Iterable[float]): Tick values.
        positions (Iterable[Sequence[float]]): Tick positions as [x, y].
        side (str): "Left" or "Right" (used for logging).
        spacing_tol (float): Relative tolerance for slope consistency.

    Returns:
        (clean_numbers, clean_positions): Lists of tick values and positions.
    """
    numbers = list(numbers)
    positions = [list(p) for p in positions]

    # Convert tick string values to float and drop any that cannot be parsed.
    valid_numbers = []
    valid_positions = []
    for v, p in zip(numbers, positions):
        try:
            valid_numbers.append(float(v))
            valid_positions.append(list(p))
        except (ValueError, TypeError):
            logger.warning(
                "Axis validation for %s side: could not convert tick value '%s' to float; skipping this tick.",
                side,
                v,
            )

    numbers = valid_numbers
    positions = valid_positions

    if len(numbers) < 2 or len(numbers) != len(positions):
        logger.warning(
            "Axis validation failed for %s side: insufficient or mismatched ticks "
            "(%d values, %d positions).",
            side,
            len(numbers),
            len(positions),
        )
        return numbers, positions

    # positions are [x, y] in image coords; sort by y (row)
    # Keep track of original indices so we can return cleaned subsets.
    indexed = list(enumerate(zip(numbers, positions)))
    indexed.sort(key=lambda t: float(t[1][1][1]))  # sort by y

    idxs = [i for i, _ in indexed]
    ordered_vals = [float(vp[0]) for _, vp in indexed]
    ys = [float(vp[1][1]) for _, vp in indexed]

    while True:
        if len(ordered_vals) < 2:
            logger.warning(
                "Axis validation warning for %s side: fewer than two ticks after filtering: %s",
                side,
                ordered_vals,
            )
            # Return whatever cleaned subset survives in idxs (possibly empty),
            # rather than the original unfiltered inputs.
            clean_numbers = [numbers[i] for i in idxs]
            clean_positions = [positions[i] for i in idxs]
            return clean_numbers, clean_positions

        # 1 - basic monotonicity check on values along the axis
        inc = all(ordered_vals[i + 1] >= ordered_vals[i] for i in range(len(ordered_vals) - 1))  # increasing or flat
        dec = all(ordered_vals[i + 1] <= ordered_vals[i] for i in range(len(ordered_vals) - 1))  # decreasing or flat


        # 2) value-vs-position consistency: check that the per-pixel slope
        #    (Δvalue / Δy) is roughly constant.
        dv = [ordered_vals[i + 1] - ordered_vals[i] for i in range(len(ordered_vals) - 1)]
        dy = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]

        # Handle duplicate positions (dy_i == 0). These mean two or more ticks share
        # exactly the same y-coordinate.
        duplicate_indices = [i for i, d in enumerate(dy) if d == 0]
        if duplicate_indices:
            # Group consecutive duplicates into clusters that share the same y.
            # Example: ys = [100, 120, 120, 120, 140]
            # dy indices with 0 might be [1, 2] → cluster of ticks [1,2,3].
            start = None
            dup_clusters = []
            for i in range(len(ys) - 1):
                if ys[i + 1] == ys[i]:
                    if start is None:
                        start = i
                else:
                    if start is not None:
                        dup_clusters.append((start, i + 1))  # inclusive start,end indices in ordered_vals
                        start = None
            if start is not None:
                dup_clusters.append((start, len(ys) - 1))

            # For each cluster of ticks at the same y:
            # - If all values identical, drop all but one (true duplicates).
            # - If values differ, drop the whole cluster (we can't know which is right).
            to_drop = set()
            for c_start, c_end in dup_clusters:
                idx_range = list(range(c_start, c_end + 1))
                vals_here = {ordered_vals[j] for j in idx_range}
                if len(vals_here) == 1:
                    # Pure duplicates: keep the last, drop the rest.
                    for j in idx_range[:-1]:
                        to_drop.add(j)
                else:
                    # Conflicting labels at same y: drop the entire cluster.
                    logger.warning(
                        "Axis validation for %s side: conflicting tick values %s at shared y=%s; "
                        "dropping all ticks at this y.",
                        side,
                        [ordered_vals[j] for j in idx_range],
                        ys[c_start],
                    )
                    for j in idx_range:
                        to_drop.add(j)

            if to_drop:
                for j in sorted(to_drop, reverse=True):
                    logger.warning(
                        "Axis validation for %s side: removing tick value %s at index %d (duplicate-y cleanup).",
                        side,
                        ordered_vals[j],
                        j,
                    )
                    ordered_vals.pop(j)
                    ys.pop(j)
                    idxs.pop(j)
                # Re-run loop with cleaned data
                continue

        slopes = [
            dv_i / dy_i for dv_i, dy_i in zip(dv, dy)
            if dy_i != 0  # skip zero-height steps
        ]

        if len(slopes) < 2:
            # Check if this is because all positions are duplicates (all dy_i == 0)
            if len(slopes) == 0 and len(ordered_vals) >= 2:
                # All tick positions have the same y-coordinate - axis is useless
                logger.warning(
                    "Axis validation for %s side: all tick positions have identical y-coordinates. "
                    "This axis appears to have duplicate positions and is unusable. Returning empty lists.",
                    side,
                )
                return [], []
            # Otherwise, not enough slopes for consistency check, but data might still be valid
            break

        # Identify the dominant slope cluster by frequency, not by mean.
        # Group slopes that are close to each other (within spacing_tol)
        # and take the largest group as the expected slope cluster.
        clusters = []  # list of (rep_slope, [indices])
        for i, s in enumerate(slopes):
            placed = False
            for ci, (rep, idxs_c) in enumerate(clusters):
                # Relative difference to cluster representative
                if rep != 0 and abs(s - rep) / abs(rep) <= spacing_tol:
                    # Merge into this cluster, update representative to simple mean
                    idxs_c.append(i)
                    clusters[ci] = (sum(slopes[j] for j in idxs_c) / len(idxs_c), idxs_c)
                    placed = True
                    break
            if not placed:
                clusters.append((s, [i]))

        if not clusters:
            break

        # Pick the cluster with the most members as the "good" cluster
        best_rep, best_idxs = max(clusters, key=lambda c: len(c[1]))

        # Any slope not in the dominant cluster is considered "bad"
        bad_indices = [i for i in range(len(slopes)) if i not in best_idxs]

        if not bad_indices:
            # All slopes look reasonably consistent
            break

        # Edge case: exactly one bad slope at the start or end – treat the
        # corresponding endpoint tick as suspect.
        if len(bad_indices) == 1:
            bi = bad_indices[0]
            if bi == 0:
                # First slope bad -> first tick is suspect
                drop_idx = 0
            elif bi == len(slopes) - 1:
                # Last slope bad -> last tick is suspect
                drop_idx = len(ordered_vals) - 1
            else:
                drop_idx = None

            if drop_idx is not None:
                logger.warning(
                    "Axis validation for %s side: removing suspect endpoint tick value %s at index %d "
                    "for validation purposes (slopes=%s, cluster_rep=%0.3f)",
                    side,
                    ordered_vals[drop_idx],
                    drop_idx,
                    slopes,
                    best_rep,
                )
                ordered_vals.pop(drop_idx)
                ys.pop(drop_idx)
                idxs.pop(drop_idx)
                continue

        # If exactly two consecutive bad slopes, drop the interior tick between them
        if len(bad_indices) == 2 and bad_indices[1] == bad_indices[0] + 1:
            drop_idx = bad_indices[0] + 1  # index of the middle tick
            logger.warning(
                "Axis validation for %s side: removing suspect tick value %s at index %d "
                "for validation purposes (slopes=%s, cluster_rep=%0.3f)",
                side,
                ordered_vals[drop_idx],
                drop_idx,
                slopes,
                best_rep,
            )
            # Remove that tick from local copies and loop again
            ordered_vals.pop(drop_idx)
            ys.pop(drop_idx)
            idxs.pop(drop_idx)
            continue

        # Special case: three bad slopes – one at an endpoint and two adjacent in the middle.
        # In this scenario we:
        #   - remove the tick between the two adjacent bad slopes, and
        #   - remove the endpoint tick indicated by the endpoint bad slope.
        if len(bad_indices) == 3:
            last_slope_idx = len(slopes) - 1
            has_start = 0 in bad_indices
            has_end = last_slope_idx in bad_indices

            # Find the interior adjacent pair (k, k+1)
            interior_pair = None
            for bi in bad_indices:
                if bi != 0 and bi != last_slope_idx and (bi + 1) in bad_indices:
                    interior_pair = (bi, bi + 1)
                    break

            if (has_start or has_end) and interior_pair is not None:
                k, k1 = interior_pair
                mid_tick = k + 1  # tick index between the two bad slopes

                # Endpoint tick to drop: first or last tick
                endpoint_tick = 0 if has_start else len(ordered_vals) - 1

                # Drop in descending index order so indices remain valid
                for drop_idx in sorted({mid_tick, endpoint_tick}, reverse=True):
                    logger.warning(
                        "Axis validation for %s side: removing suspect tick value %s at index %d "
                        "(three-bad-slopes pattern, slopes=%s, cluster_rep=%0.3f)",
                        side,
                        ordered_vals[drop_idx],
                        drop_idx,
                        slopes,
                        best_rep,
                    )
                    ordered_vals.pop(drop_idx)
                    ys.pop(drop_idx)
                    idxs.pop(drop_idx)
                continue

        # If we reach here with remaining bad slopes, we can't safely decide which
        # tick(s) to drop without risking over-aggressive correction.
        logger.warning(
            "Axis validation warning for %s side: inconsistent value-per-pixel slopes "
            "(cluster_rep=%0.3f, slopes=%s); unable to unambiguously identify bad ticks.",
            side,
            best_rep,
            slopes,
        )
        break

    # Reconstruct cleaned numbers/positions from surviving indices
    clean_numbers = [numbers[i] for i in idxs]
    clean_positions = [positions[i] for i in idxs]
    return clean_numbers, clean_positions


def validate_axis_pair(
    left_numbers,
    left_positions,
    right_numbers,
    right_positions,
    y_tol_pixels: float = 5.0,
):
    """
    Cross-check consistency between left and right axes.

    For any tick value that appears on both sides, we expect the corresponding
    y-coordinates to be approximately equal (within `y_tol_pixels`). If not,
    a warning is logged.

    This function does not mutate any inputs; it is purely diagnostic.

    Args:
        left_numbers (Iterable[float]): Tick values on the left axis.
        left_positions (Iterable[Sequence[float]]): Left positions as [x, y].
        right_numbers (Iterable[float]): Tick values on the right axis.
        right_positions (Iterable[Sequence[float]]): Right positions as [x, y].
        y_tol_pixels (float): Allowed absolute difference in y between matching
                              tick values on the two sides.

    Returns:
        bool: True if all matching ticks are within tolerance, False otherwise.
    """
    left_numbers = list(left_numbers)
    right_numbers = list(right_numbers)
    left_positions = [list(p) for p in left_positions]
    right_positions = [list(p) for p in right_positions]

    # Build value -> list of y-coordinates maps for each side
    left_map = {}
    for v, p in zip(left_numbers, left_positions):
        left_map.setdefault(float(v), []).append(float(p[1]))

    right_map = {}
    for v, p in zip(right_numbers, right_positions):
        right_map.setdefault(float(v), []).append(float(p[1]))

    common_vals = sorted(set(left_map.keys()) & set(right_map.keys()))
    if not common_vals:
        # Nothing to compare; treat as inconclusive but not a failure.
        return True

    ok = True
    for v in common_vals:
        # Compare mean y for this tick value on each side
        y_left = sum(left_map[v]) / len(left_map[v])
        y_right = sum(right_map[v]) / len(right_map[v])
        if abs(y_left - y_right) > y_tol_pixels:
            logger.warning(
                "Axis pair validation warning: tick value %s has inconsistent y "
                "between sides (left=%0.2f, right=%0.2f, tol=%0.2f).",
                v,
                y_left,
                y_right,
                y_tol_pixels,
            )
            ok = False

    return ok


def estimate_zero_line_y_axis(numbers, positions, eps: float = 1e-3):
    """
    Estimate the y-coordinate of the value 0 line for a single axis, based on
    tick values and their positions.

    Strategy:
      1. If any tick has value exactly (or very close to) 0, return the mean
         y of those ticks.
      2. Otherwise, look for neighbouring ticks whose values straddle 0
         (one negative, one positive) and linearly interpolate y at value 0.
         If multiple candidates exist, choose the pair with smallest
         |v_neg| + |v_pos|.

    Args:
        numbers (Iterable[float]): Tick values.
        positions (Iterable[Sequence[float]]): Tick positions as [x, y].
        eps (float): Tolerance for treating a tick value as zero.

    Returns:
        float or None: Estimated y-coordinate of the 0-line for this axis,
                       or None if it cannot be estimated.
    """
    numbers = [float(v) for v in numbers]
    positions = [list(p) for p in positions]
    if len(numbers) != len(positions) or not numbers:
        return None

    # Sort by y (row) so we move along the axis consistently
    vals = [(v, float(p[1])) for v, p in zip(numbers, positions)]
    vals.sort(key=lambda t: t[1])
    vs = [v for v, _ in vals]
    ys = [y for _, y in vals]

    # 1) Exact (or near) zero tick(s)
    # If any tick is very close to value 0 (within eps), we treat its y-position
    # as lying on the physical zero line. Multiple such ticks are averaged.
    zero_ys = [y for v, y in zip(vs, ys) if abs(v) <= eps]
    if zero_ys:
        return sum(zero_ys) / len(zero_ys)

    # 2) No explicit 0-tick: fit a straight line y = a * v + b through all
    #    (value, y) pairs, and evaluate it at v = 0. This does *not* require
    #    both positive and negative values: with at least two distinct tick
    #    values we can still extrapolate the mapping down to 0.
    if len(vs) < 2:
        # Not enough distinct information to define a line
        return None

    try:
        # np.polyfit returns coefficients [a, b] for y ≈ a*v + b
        a, b = np.polyfit(vs, ys, 1)
        # At v = 0 we have y0 = b
        return float(b)
    except Exception:
        # In degenerate cases (e.g. numerical issues), give up gracefully
        return None


def estimate_zero_line_y(
    left_numbers=None,
    left_positions=None,
    right_numbers=None,
    right_positions=None,
    y_tol_pixels: float = 5.0,
):
    """
    Estimate a single global y-coordinate for the value 0 line, using one or
    both axes if available.

    - If both left and right estimates are available and within `y_tol_pixels`,
      their average is returned.
    - If both are available but differ more than `y_tol_pixels`, their average
      is still returned, but a warning is logged.
    - If only one side yields an estimate, that value is returned.
    - If neither side yields an estimate, returns None.

    Additionally, when one axis has very few ticks (<= 2) and the other has
    more, the side with *more* ticks is trusted preferentially, because with
    only two ticks there is only a single slope and no redundancy for
    consistency checks.

    Args:
        left_numbers, left_positions: As for `estimate_zero_line_y_axis`.
        right_numbers, right_positions: As for `estimate_zero_line_y_axis`.
        y_tol_pixels (float): Tolerance for considering left/right y0 equal.

    Returns:
        float or None: Estimated y-coordinate of the 0-line, or None.
    """
    # Initialise estimates for left and right axes
    yL = None
    yR = None

    # Track how many ticks each axis actually has, so we can prefer the
    # side with more information if necessary.
    nL = len(left_numbers) if left_numbers is not None else 0
    nR = len(right_numbers) if right_numbers is not None else 0

    # Minimum number of ticks considered reliable for zero-line estimation.
    # With 2 or fewer ticks there is only a single slope and no redundancy.
    MIN_RELIABLE_TICKS = 3

    # Compute per-axis zero-line estimates when input is available.
    if left_numbers is not None and left_positions is not None:
        yL = estimate_zero_line_y_axis(left_numbers, left_positions)

    if right_numbers is not None and right_positions is not None:
        yR = estimate_zero_line_y_axis(right_numbers, right_positions)

    # If neither side produced an estimate, cannot infer a global zero-line.
    if yL is None and yR is None:
        return None

    # If only one side has an estimate, return it directly.
    if yL is None:
        return yR
    if yR is None:
        return yL

    # At this point both sides produced an estimate. If one axis has very few
    # ticks and the other has "enough", prefer the larger axis instead of
    # averaging. For example, if the right axis only returned 2 labels while
    # the left returned 3, trust the left-hand estimate.
    if nL >= MIN_RELIABLE_TICKS and nR < MIN_RELIABLE_TICKS:
        return yL
    if nR >= MIN_RELIABLE_TICKS and nL < MIN_RELIABLE_TICKS:
        return yR

    # Otherwise both sides are comparable in terms of tick count. If they
    # disagree by more than the allowed tolerance, log a warning but still
    # use their average as a compromise.
    if abs(yL - yR) > y_tol_pixels:
        logger.warning(
            "Zero-line y estimate disagrees between sides (left=%0.2f, right=%0.2f, tol=%0.2f); "
            "using their average.",
            yL,
            yR,
            y_tol_pixels,
        )

    # Return the mean of the two side estimates.
    return 0.5 * (yL + yR)


def plot_digitized_data(Rticks, Rlocs, Lticks, Llocs, top_curve_coords):
    """
    Digitize and plot the data.

    This function digitizes and plots the segmented data based on tick marks from the left 
    and right axes and the top curve coordinates. It aligns the data with the 
    given tick marks, inverts the waveform if necessary, and scales the data 
    to an arbitrary time scale and flow rate. The plot is then generated with 
    the x-axis representing the arbitrary time scale and the y-axis showing 
    the flow rate.

    Args:
        Rticks (list): A list of tick values on the right axis, assumed to be 
            in ascending order.
        Rlocs (list): A list of the locations (x, y coordinates) for each 
            tick on the right axis.
        Lticks (list): A list of tick values on the left axis, assumed to be 
            in ascending order.
        Llocs (list): A list of the locations (x, y coordinates) for each 
            tick on the left axis.
        top_curve_coords (list): A list of (x, y) coordinates representing 
            the top curve of the waveform to be digitized.

    Returns:
        (tuple) : tuple containing:
            - **Xplot** (list): A list of x-values for the digitized data, scaled to an arbitrary time scale.
            - **Yplot** (list): A list of y-values for the digitized data, representing flow rate in cm/s.
            - **Ynought** (list): A list containing the 0 value y co-ordinate for the digitized data (not used).

    Note:
        The function adjusts for cases where there is only one tick mark on 
        the right axis, ensuring the digitization process can proceed. It 
        also inverts the waveform if the average flow rate is negative.
    """

    # Handle empty axis data (e.g., from duplicate positions)
    if not Lticks or not Rticks:
        if not Lticks and not Rticks:
            logger.error("Both axes are empty - cannot digitize")
            return [], [], [0]
        # If one axis is empty, we can still proceed with the other
        if not Lticks:
            logger.warning("Left axis is empty - using right axis only")
            # Use right axis for both (not ideal but allows processing to continue)
            Lticks = Rticks.copy()
            Llocs = Rlocs.copy()
        elif not Rticks:
            logger.warning("Right axis is empty - using left axis only")
            Rticks = Lticks.copy()
            Rlocs = Llocs.copy()

    # We will have problems if the right axes only has 1 tick and that tick is equal to the minimum on the left axis
    if len(Rticks) == 1:
        Rticks.append(Lticks[-1])
        Rlocs.append([Rlocs[-1][0], Llocs[-1][1]])

    Rticks = list(map(int, Rticks))
    XmaxtickR = max(Rticks)
    XmaxidR = Rticks.index(XmaxtickR)
    XmaxR = Rlocs[XmaxidR][0]
    YmaxR = Rlocs[XmaxidR][1]
    XMintickR = min(Rticks)
    XMinidR = Rticks.index(XMintickR)
    XminR = Rlocs[XMinidR][0]
    #
    Lticks = list(map(int, Lticks))
    XmaxtickL = max(Lticks)
    XmaxidL = Lticks.index(XmaxtickL)
    XmaxL = Llocs[XmaxidL][0]
    XMintickL = min(Lticks)
    XminidL = Lticks.index(XMintickL)
    XminL = Llocs[XminidL][0]
    YminL = Llocs[XminidL][1]

    # Yplots = [Llocs[XmaxidL][0], Llocs[XminidL][0], Rlocs[XmaxidR][0]]
    # Xplots = [Llocs[XmaxidL][1], Llocs[XminidL][1], Rlocs[XmaxidR][1]]

    Xmin = 0
    Xmax = 1
    Ymin = XMintickL
    Ymax = XmaxtickL

    b = top_curve_coords
    b = sorted(b, key=lambda k: [k[1], k[0]])

    b = [B.tolist() for B in b]
    b = [x[::-1] for x in b]

    b = pd.DataFrame(b).groupby(0, as_index=False)[1].mean().values.tolist()
    b = [x[::-1] for x in b]

    X = [XminL, XmaxR]
    Y = [YminL, YmaxR]

    for i in range(0, len(b)):
        X.append(b[i][1])
        Y.append(b[i][0])

    origin = [X[0], Y[0]]
    topRight = [X[1], Y[1]]
    XminScale = origin[0]
    XmaxScale = topRight[0]
    YminScale = origin[1]
    YmaxScale = topRight[1]

    Ynought = [(0 - YminScale) / (YmaxScale - YminScale) * (Ymax - Ymin) + Ymin]

    X = X[2:-1]
    Y = Y[2:-1]

    Xplot = [
        (i - XminScale) / (XmaxScale - XminScale) * (Xmax - Xmin) + Xmin for i in X
    ]
    Yplot = [
        (i - YminScale) / (YmaxScale - YminScale) * (Ymax - Ymin) + Ymin for i in Y
    ]

    # Inverts the waveform if need be
    if np.mean(Yplot) < 0:
        Yplot = [y * (-1) for y in Yplot]

    plt.figure(2)
    plt.plot(Xplot, Yplot, "-")
    plt.xlabel("Arbitrary time scale")
    plt.ylabel("Flowrate (cm/s)")
    return Xplot, Yplot, Ynought


def plot_digitized_data_single_axis(Rticks, Rlocs, Lticks, Llocs, top_curve_coords):
    """
    Simplified digitization using a single vertical axis and a normalized X axis.

    - Chooses one "best" vertical axis (Left or Right) for value calibration.
      If both sides are present and broadly agree, uses the side with more ticks.
      If only one side is usable, falls back to that side.
    - Uses that axis to build a linear mapping from pixel y to physical value.
    - Uses curve x-position (or index) to build an arbitrary normalized time axis [0, 1].

    Args:
        Rticks, Rlocs, Lticks, Llocs, top_curve_coords: as for ``plot_digitized_data``.

    Returns:
        Xplot, Yplot, Ynought: same semantics as ``plot_digitized_data``.
    """

    # Convert to lists defensively
    Rticks = list(Rticks) if Rticks is not None else []
    Rlocs = [list(p) for p in Rlocs] if Rlocs is not None else []
    Lticks = list(Lticks) if Lticks is not None else []
    Llocs = [list(p) for p in Llocs] if Llocs is not None else []

    # If both sides are completely empty, we cannot digitize
    if not Rticks and not Lticks:
        logger.error("Digitization: both axes empty - cannot digitize waveform.")
        return [], [], [0]

    # Helper: decide which axis to use for calibration
    def choose_axis():
        has_left = len(Lticks) >= 2 and len(Lticks) == len(Llocs)
        has_right = len(Rticks) >= 2 and len(Rticks) == len(Rlocs)

        if not has_left and not has_right:
            # Not enough information on either side
            logger.error(
                "Digitization: insufficient ticks on both axes "
                "(Left: %d, Right: %d).",
                len(Lticks),
                len(Rticks),
            )
            return None, None

        if has_left and has_right:
            # Check rough agreement between sides
            try:
                agree = validate_axis_pair(Lticks, Llocs, Rticks, Rlocs)
            except Exception:
                traceback.print_exc()
                agree = False

            # Prefer the side with more ticks if they agree, otherwise still pick
            # the denser side but log a warning.
            if agree:
                if len(Lticks) >= len(Rticks):
                    return Lticks, Llocs
                return Rticks, Rlocs
            else:
                logger.warning(
                    "Digitization: left/right axes disagree; using the side with more ticks "
                    "(Left: %d, Right: %d).",
                    len(Lticks),
                    len(Rticks),
                )
                if len(Lticks) >= len(Rticks):
                    return Lticks, Llocs
                return Rticks, Rlocs

        # Only one side has usable data
        if has_left:
            logger.info("Digitization: using left axis only for calibration.")
            return Lticks, Llocs
        else:
            logger.info("Digitization: using right axis only for calibration.")
            return Rticks, Rlocs

    axis_ticks, axis_locs = choose_axis()
    if axis_ticks is None or axis_locs is None:
        # Already logged
        return [], [], [0]

    # Build a simple linear mapping from pixel y to value using the chosen axis
    # Axis locations are [x, y]; we care about y here.
    try:
        pairs = [(float(p[1]), float(v)) for v, p in zip(axis_ticks, axis_locs)]
    except Exception:
        traceback.print_exc()
        logger.error("Digitization: failed to build (y, value) pairs from axis data.")
        return [], [], [0]

    # Sort by y (image coordinates)
    pairs.sort(key=lambda t: t[0])
    ys_axis = [t[0] for t in pairs]
    vals_axis = [t[1] for t in pairs]

    if len(ys_axis) < 2 or ys_axis[0] == ys_axis[-1]:
        logger.error(
            "Digitization: axis has insufficient vertical spread for calibration "
            "(ys=%s).",
            ys_axis,
        )
        return [], [], [0]

    # End-point linear calibration: value = a * y + b
    a = (vals_axis[-1] - vals_axis[0]) / (ys_axis[-1] - ys_axis[0])
    b = vals_axis[0] - a * ys_axis[0]

    # Process curve coordinates: one point per column, as in the original implementation
    b_coords = top_curve_coords
    if b_coords is None or len(b_coords) == 0:
        logger.error("Digitization: top_curve_coords is empty - nothing to digitize.")
        return [], [], [0]

    # Sort by (x, y) and average rows per column
    b_arr = [list(B) for B in b_coords]
    # b_arr is [row, col]; we want [col, row] for grouping by x
    b_swapped = [x[::-1] for x in b_arr]
    df = pd.DataFrame(b_swapped).groupby(0, as_index=False)[1].mean().values.tolist()
    # Back to [row, col]
    b_clean = [x[::-1] for x in df]

    # Build X (pixels) and Y (pixels) from the cleaned curve
    X_pixels = [pt[1] for pt in b_clean]
    Y_pixels = [pt[0] for pt in b_clean]

    if len(X_pixels) < 2:
        logger.error("Digitization: insufficient curve points after cleaning.")
        return [], [], [0]

    # Normalize X to [0, 1] as arbitrary time axis
    Xmin_pix = min(X_pixels)
    Xmax_pix = max(X_pixels)
    if Xmax_pix == Xmin_pix:
        Xplot = [0.0 for _ in X_pixels]
    else:
        Xplot = [(x - Xmin_pix) / (Xmax_pix - Xmin_pix) for x in X_pixels]

    # Map pixel y to physical value using the axis calibration
    Yplot = [a * y + b for y in Y_pixels]

    # Invert waveform if mean is negative (same heuristic as original)
    if np.mean(Yplot) < 0:
        Yplot = [y * (-1) for y in Yplot]

    # In this simplified scheme, y=0 in digitized space is just 0
    Ynought = [0.0]

    plt.figure(2)
    plt.plot(Xplot, Yplot, "-")
    plt.xlabel("Arbitrary time scale")
    plt.ylabel("Flowrate (cm/s)")

    return Xplot, Yplot, Ynought


def plot_digitized_data_dicom(dicom_metadata, top_curve_coords=None):
    """
    Digitize waveform for DICOM using metadata. Uses the same curve ordering as
    plot_digitized_data_single_axis: one point per column, sorted left-to-right,
    so plt.plot(Xplot, Yplot) draws a proper waveform.
    """
    Xplot, Yplot = [], []
    Ynought = [float(dicom_metadata.get("ReferencePixelPhysicalValueY", 0.0))]

    if top_curve_coords is None or len(top_curve_coords) == 0:
        return Xplot, Yplot, Ynought

    # --- Same ordering as single_axis: curve is [row, col], group by column, one point per column ---
    # top_curve_coords: list of [row, col] (y_pixel, x_pixel) in arbitrary order
    b_arr = [list(pt) for pt in top_curve_coords]
    # Swap to [col, row] so we can group by column (index 0)
    b_swapped = [pt[::-1] for pt in b_arr]
    # One point per column: mean row (y) per column (x). Result rows stay in column order.
    grouped = pd.DataFrame(b_swapped).groupby(0, as_index=False)[1].mean().values.tolist()
    # Back to [row, col]; now sorted by column so we go left-to-right
    b_clean = [pt[::-1] for pt in grouped]
    # Optional: sort by column so X is strictly increasing (groupby may not guarantee order)
    b_clean.sort(key=lambda pt: pt[1])

    col_pixels = [pt[1] for pt in b_clean]   # x in image
    row_pixels = [pt[0] for pt in b_clean]  # y in image

    # --- Map pixel (col, row) to physical (X, Y) using DICOM metadata ---
    min_x = dicom_metadata.get("RegionLocationMinX0")
    min_y = dicom_metadata.get("RegionLocationMinY0")
    ref_x0 = dicom_metadata.get("ReferencePixelX0", 0)
    ref_y0 = dicom_metadata.get("ReferencePixelY0", 0)
    x_ref = min_x + ref_x0
    y_ref = min_y + ref_y0
    x_ref_phys = float(dicom_metadata.get("ReferencePixelPhysicalValueX", 0.0))
    y_ref_phys = float(dicom_metadata.get("ReferencePixelPhysicalValueY", 0.0))
    dx = float(dicom_metadata.get("PhysicalDeltaX", 1.0))
    dy = float(dicom_metadata.get("PhysicalDeltaY", 1.0))
    dy = -abs(dy)

    col_pixels = np.asarray(col_pixels, dtype=float)
    row_pixels = np.asarray(row_pixels, dtype=float)

    Xplot = x_ref_phys + (col_pixels - x_ref) * dx
    Yplot = y_ref_phys + (row_pixels - y_ref) * dy

    plt.plot(Xplot, Yplot, "-")
    plt.xlabel("Physical X (time or distance)")
    plt.ylabel("Physical Y (e.g. velocity)")
    return Xplot, Yplot, Ynought


def plot_correction(Xplot, Yplot, df):
    """
    Adjusts and corrects the digitized waveform data using extracted text data, identifies
    and filters peaks and troughs, computes hemodynamic parameters, scales the time axis,
    and plots the corrected waveform.

    The function works by first inserting a new column for digitized values in the DataFrame.
    It processes the waveform to identify and filter peaks and troughs, computes key 
    hemodynamic parameters (PS, ED, S/D, RI, TAmax, PI), and updates these values in the
    DataFrame. It also scales the x-axis to real-time based on heart rate information and
    generates a plot of the corrected waveform.

    Exceptions are handled by printing traceback information.

    Args:
        Xplot (list of float): The x-coordinates (time axis) of the waveform data.
        Yplot (list of float): The y-coordinates (flowrate axis) of the waveform data.
        df (pandas.DataFrame): The DataFrame with extracted text data including 
                               hemodynamic parameters and heart rate.

    Returns:
        **df** (pandas.DataFrame): The DataFrame updated with computed parameters in the "Digitized Value" column.
    """

    try:
        # import Jinja2
        y = np.array(Yplot)
        x = np.array(Xplot)
        # DF = df
        df.insert(loc=3, column="Digitized Value", value="")
        peaks, _ = find_peaks(y)  # PSV
        troughs, _ = find_peaks(-y)  # EDV
        # filter out any anomalous signals:
        trough_widths, _, _, _ = peak_widths(-y, troughs)
        mean_widths_troughs = statistics.mean(trough_widths)
        # filter out anomalous peaks
        valid_troughs = []
        for i in range(len(troughs)):
            if trough_widths[i] > (mean_widths_troughs / 2):
                valid_troughs.append(troughs[i])
        troughs = valid_troughs

        results_half = peak_widths(y, peaks)
        widths_of_peaks = results_half[0]
        mean_widths_peaks = statistics.mean(widths_of_peaks)
        valid_peaks = []
        for i in range(len(peaks)):
            if widths_of_peaks[i] > (mean_widths_peaks / 2) and y[peaks[i]] > (
                    statistics.mean(y[peaks]) * 0.8
            ):
                valid_peaks.append(peaks[i])
        peaks = valid_peaks

        # Peak systolic
        PS = statistics.mean(y[peaks])
        # End diastolic
        ED = statistics.mean(y[troughs])
        # Find S/D
        SoverD = statistics.mean(y[peaks]) / statistics.mean(y[troughs])
        # Find RI
        RI = (
                     statistics.mean(y[peaks]) - statistics.mean(y[troughs])
             ) / statistics.mean(y[peaks])
        # Find TAmax
        TAmax = (statistics.mean(y[peaks]) + (2 * statistics.mean(y[troughs]))) / 3
        # Find PI
        PI = (
                     statistics.mean(y[peaks]) - statistics.mean(y[troughs])
             ) / statistics.mean(y)

        words = ["PS", "ED", "S/D", "RI", "TA"]
        values = [PS, ED, SoverD, RI, TAmax]
        values = [round(elem, 2) for elem in values]
        # Loop through each word and value
        for i in range(len(words)):
            word = words[i]
            value = values[i]
            try:
                # Find rows where word is in "Text" column, and set corresponding "Value" to the value
                df.loc[df["Word"].str.contains(word), "Digitized Value"] = value

                # get_colour(df.loc[df["Word"].str.contains(word),'Value'].values[0],value)
            except Exception:
                continue

        # Period of the signal in arbitrary scale
        arbitrary_period = (x[peaks[-1]] - x[peaks[1]]) / (len(peaks) - 2)
    except Exception:
        traceback.print_exc()  # prints the error message and traceback

    try:
        # HR in bpm from extracted text
        hr_vals = df.loc[df["Word"].str.contains("HR"), "Value"].values
        hr = float(hr_vals[0]) if len(hr_vals) > 0 else 0.0

        # If HR is zero or invalid, skip time scaling 
        if not np.isfinite(hr) or hr == 0.0:
            logger.warning("Invalid HR value for time scaling (%s); skipping time scaling plot.", hr)
            return df

        # Period of the signal in real time scale from text extraction
        real_period = 1.0 / (hr / 60.0)
        # Calculate a scaling factor
        scale_factor = real_period / arbitrary_period
        x_time = x * scale_factor
        plt.close(2)
        plt.figure(2)
        plt.plot(x_time, y)
        plt.plot(x_time[peaks], y[peaks], "x")
        plt.plot(x_time[troughs], y[troughs], "x")
        # plt.plot(x_time[troughs[1:-1]],y[troughs[1:-1]],"x")
        # Check if any value in x_time is NaN or zero
        if not any(np.isnan(x) or x == 0 or np.isinf(x) for x in x_time):
            plt.xlim((min(x_time), max(x_time)))
        plt.ylim((0, max(y) + 10))
        plt.xlabel("Time (s)")
        plt.ylabel("Flowrate (cm/s)")
    except Exception:
        logger.warning("Could not plot time-scaled waveform; continuing.", exc_info=True)

    return df


def scan_type_test(input_image_filename):
    """
    Function for yellow filtering an image and searching for a list of target words indicative of
    a doppler ultrasound scan taken using the Voluson E8, RAB6-D.

    Args:
        input_image_filename (str) : Name of file within current directory, or path to a file.

    Returns:
        **Fail** (int) : Idicates if the file is a fail (1) - doesn't meet criteria for a doppler ultrasound, or pass (0) - does meet criteria. 

    """

    img = cv2.imread(input_image_filename)  # Input image file
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert to HSV
    lower_yellow = np.array([1, 100, 100], dtype=np.uint8)  # Lower yellow bound
    upper_yellow = np.array([200, 255, 255], dtype=np.uint8)  # Upper yellow bound
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)  # Threshold HSV between bounds
    yellow_text = cv2.bitwise_and(gray, gray, mask=mask)

    yellow_text[int(img.shape[1] * 0.45): img.shape[1], :] = 0  # Exclude bottom 3rd of image - target scans have no text of interest here.
    pixels = np.array(yellow_text)
    data = pytesseract.image_to_data(
        pixels, output_type=Output.DICT, lang="eng", config="--psm 3 "
    )

    # Loop through each word and draw a box around it
    for i in range(len(data["text"])):
        x = data["left"][i]
        y = data["top"][i]
        segmentation_mask = data["width"][i]
        h = data["height"][i]
        if int(data["conf"][i]) > 1:
            cv2.rectangle(img, (x, y), (x + segmentation_mask, y + h), (0, 0, 255), 2)

    # Display image
    # cv2.imshow('img', img)

    # Perform OCR on the preprocessed image
    custom_config = r"--oem 3 --psm 3"
    text = pytesseract.image_to_string(pixels, lang="eng", config=custom_config)

    # Analyze the OCR output
    lines = text.splitlines()
    target_words = [
        "HR",
        "TAmax",
        "Lt Ut-PS",
        "Lt Ut-ED",
        "Lt Ut-S/D",
        "Lt Ut-PI",
        "Lt Ut-RI",
        "Lt Ut-MD",
        "Lt UT-TAmax",
        "Lt Ut-HR",
        "Rt Ut-PS",
        "Rt Ut-ED",
        "Rt Ut-S/D",
        "Rt Ut-PI",
        "Rt Ut-RI",
        "Rt Ut-MD",
        "Rt UT-TAmax",
        "Rt Ut-HR",
        "Umb-PS",
        "Umb-ED",
        "Umb-S/D",
        "Umb-PI",
        "Umb-RI",
        "Umb-MD",
        "Umb-TAmax",
        "Umb-HR",
    ]  # Target words to search for - there may be more to add to this.

    # Split text into lines
    lines = text.split("\n")

    # Initialize DataFrame
    df = pd.DataFrame(columns=["Line", "Word", "Value", "Unit"])

    Fail = 1  # initialise fail variable
    for target in target_words:
        for word in lines:
            if target in word:
                Fail = 0  # If any of the words are found, then no fail.

    return Fail, df  # Return the fail variable and dataframe contraining extracted text.


def annotate(
        input_image_obj,
        refined_segmentation_mask,
        Left_dimensions,
        Right_dimensions,
        Waveform_dimensions,
        Left_axis,
        Right_axis,
):
    """
    Visual aid for evaluating segmentation.

    Annotates the original image with the computed components from each of the
    previous functions by overlaying polygons on the regions of interest (ROIs)
    and highlighting them in specific colors.

    The function draws perimeters around the provided dimensions for the left,
    right, and waveform components on the segmentation mask. Then, it modifies
    the input image's pixel data to overlay these perimeters and color-codes the
    different regions: the waveform in red and the left/right axes in green,
    and the bounds of the ticks and labels in magenta.

    Args:
        input_image_obj (PIL.Image.Image): The original image to be annotated.
        refined_segmentation_mask (numpy.ndarray): The segmentation mask that
            indicates the regions of interest.
        Left_dimensions (tuple): The (x_min, x_max, y_min, y_max) dimensions for
            the left region of interest.
        Right_dimensions (tuple): The (x_min, x_max, y_min, y_max) dimensions for
            the right region of interest.
        Waveform_dimensions (tuple): The (x_min, x_max, y_min, y_max) dimensions
            for the waveform region.
        Left_axis (numpy.ndarray): The segmentation mask (ticks and labels) for the left axis.
        Right_axis (numpy.ndarray): The segmentation mask (ticks and labels) for the right axis.

    Returns:
        PIL.Image.Image: The annotated image with ROIs color-coded and highlighted.
    """

    Xmin, Xmax, Ymin, Ymax = Waveform_dimensions
    Xmin_R, Xmax_R, Ymin_R, Ymax_R = Right_dimensions
    Xmin_L, Xmax_L, Ymin_L, Ymax_L = Left_dimensions

    rR = [Xmin_R, Xmax_R, Xmax_R, Xmin_R, Xmin_R]
    cR = [Ymax_R, Ymax_R, Ymin_R, Ymin_R, Ymax_R]
    rrR, ccR = polygon_perimeter(cR, rR, refined_segmentation_mask.shape)

    rL = [Xmin_L, Xmax_L, Xmax_L, Xmin_L, Xmin_L]
    cL = [Ymax_L, Ymax_L, Ymin_L, Ymin_L, Ymax_L]
    rrL, ccL = polygon_perimeter(cL, rL, refined_segmentation_mask.shape)

    r = [Xmin, Xmax, Xmax, Xmin, Xmin]
    c = [Ymax, Ymax, Ymin, Ymin, Ymax]
    rr, cc = polygon_perimeter(c, r, refined_segmentation_mask.shape)

    refined_segmentation_mask[rr, cc] = 2  # set color white
    refined_segmentation_mask[rrL, ccL] = 2
    refined_segmentation_mask[rrR, ccR] = 2

    img_RGB = input_image_obj  # .convert('RGB')
    pixel_data = img_RGB.load()
    # gry = img_RGB.convert("L")  # returns grayscale version.

    for y in range(img_RGB.size[1]):
        for x in range(img_RGB.size[0]):
            # r = pixel_data[x, y][0]
            # g = pixel_data[x, y][1]
            # b = pixel_data[x, y][2]
            # rgb_values = [r, g, b]
            # min_rgb = min(rgb_values)
            # max_rgb = max(rgb_values)
            # rgb_range = max_rgb - min_rgb

            if refined_segmentation_mask[y, x] == 1:
                pixel_data[x, y] = (
                    255,
                    pixel_data[x, y][1],
                    pixel_data[x, y][2],
                    250,
                )  # Segmented waveform as Red
            elif refined_segmentation_mask[y, x] == 2:
                pixel_data[x, y] = (1, 255, 1, 255)  # Set ROIs to blue
            elif Left_axis[y, x] == 255:
                pixel_data[x, y] = (255, 0, 255, 255)  # Set ROIs to blue
            elif Right_axis[y, x] == 255:
                pixel_data[x, y] = (255, 0, 255, 255)  # Set ROIs to blue
            else:
                pixel_data[x, y] = pixel_data[x, y]
    return img_RGB


def colour_extract(input_image_obj, TargetRGB, cyl_length, cyl_radius):
    """Extract target colours from image.
    
    **The `Colour_extract()` function is deprecated and has been replaced by the function `Colour_extract_vectorised()`.**
    **See the documentation for `Colour_extract_vectorised()` for more information.**

    Args:
        input_image_obj (PIL Image) : PIL Image object.
        TargetRGB (list) : triplet of target Red Green Blue colour [Red,Green,Blue].
        cyl_length (int) : length of cylinder.
        cyl_radius (int) : radius of cylinder.

    Returns:
        COL (JpegImageFile) : PIL JpegImageFile of the filtered image highlighting selected text.
    """

    # Finds the minimum and maximum magnitudes for the colour vector
    img = np.array(input_image_obj)[:, :, :3].astype(float)
    rgb_vec = np.array(TargetRGB)
    mag = np.sqrt(np.sum(rgb_vec ** 2))
    img_dot = np.zeros_like(img[:, :, 0])

    for channel, val in enumerate(rgb_vec):
        img_dot += img[:, :, channel] * val
    img_dot /= mag

    min_mag = mag - cyl_length / 2
    max_mag = mag + cyl_length / 2

    # Finds the distance of the colours to the cylinder axis
    img_cross = np.zeros_like(img_dot)
    for channel, val in enumerate(rgb_vec):
        c1 = channel + 1 if channel + 1 < len(rgb_vec) else channel + 1 - len(rgb_vec)
        c2 = channel + 2 if channel + 2 < len(rgb_vec) else channel + 2 - len(rgb_vec)
        img_cross += (rgb_vec[c1] * img[:, :, c2] - rgb_vec[c2] * img[:, :, c1]) ** 2
    distance_to_axis = np.sqrt(img_cross) / mag

    mask = np.logical_and(
        np.logical_and(img_dot <= max_mag, img_dot >= min_mag),
        distance_to_axis <= cyl_radius,
    ).astype(np.uint8) * 255

    return Image.fromarray(mask).convert("RGB")


def append_spherical_1point(xyz):
    """
    Appends spherical coordinates to a 3D point in Cartesian coordinates.

    This function takes a single point (xyz) in Cartesian coordinates and calculates
    its corresponding spherical coordinates. It appends the radial distance from the origin (r),
    the angle in the XY plane from the positive X-axis (theta), and the angle from the
    positive Z-axis (alpha), all in degrees.

    Args:
        xyz (np.ndarray): A numpy array of shape (3,) representing the x, y, and z 
                          Cartesian coordinates of a point.

    Returns:
        **ptsnew** (ndarray): The input array with the spherical coordinates appended, resulting in a numpy array of shape (7,).
    """
    ptsnew = np.hstack(
        (xyz, np.zeros(xyz.shape), np.zeros(xyz.shape), np.zeros(xyz.shape))
    )
    xy = xyz[0] ** 2 + xyz[1] ** 2
    ptsnew[3] = np.sqrt(xy)  # xy length
    ptsnew[4] = np.sqrt(xy + xyz[2] ** 2)  # magnitude of vector (radius)
    ptsnew[5] = np.arctan(np.divide(ptsnew[1], ptsnew[0])) * (180 / math.pi)  # theta
    ptsnew[6] = np.arcsin(np.divide(ptsnew[2], ptsnew[4])) * (180 / math.pi)  # alpha
    return ptsnew


def colour_extract_vectorized(input_image_obj, target_rgb, cyl_length, cyl_radius):
    """
    Extracts a specified color from an image using a cylindrical filter in RGB space.

    Given an image object and a target RGB color, this function creates a cylindrical
    filter in RGB space defined by the specified length and radius. It extracts regions
    of the image that match the color within the defined cylindrical space.

    Args:
        input_image_obj (np.ndarray): A 3D numpy array representing the RGB image.
        target_rgb (list): A list of three integers representing the target RGB color.
        cyl_length (int): The length of the cylindrical filter along the axis of the color
                          in RGB space.
        cyl_radius (int): The radius of the cylindrical filter in RGB space.

    Returns:
        output_image (PIL.Image.Image): An image where regions matching the target color are highlighted
                         and the rest of the image is set to black.
    """
    # Convert the target RGB color to spherical coordinates
    targ = np.array(target_rgb)
    out2 = append_spherical_1point(targ)

    # Calculate the coordinates of the cylinder in RGB space
    H2 = cyl_length
    O2 = math.sin(math.radians(out2[6])) * H2
    A2 = math.cos(math.radians(out2[6])) * H2
    O1 = math.sin(math.radians(out2[5])) * A2
    A1 = math.cos(math.radians(out2[5])) * A2

    # Define the start and end points of the cylindrical filter in RGB space
    R1 = out2[0] - A1
    G1 = out2[1] - O1
    B1 = out2[2] - O2
    R2 = out2[0] + A1
    G2 = out2[1] + O1
    B2 = out2[2] + O2
    start = np.array([R1, G1, B1])
    end = np.array([R2, G2, B2])
    r = cyl_radius

    # Convert the image to a numpy array (only considering RGB channels)
    img_array = input_image_obj

    # Vectorized computation to check if each pixel lies within the cylinder in RGB space
    # Calculate the vector defining the cylinder axis
    vec = end - start
    # Calculate the cylinder's constraint based on its radius
    constraint = r * np.linalg.norm(vec)
    # Calculate the cross products for each pixel in the image
    cross_products = np.cross(img_array - start, vec)
    # Calculate the dot products for the start and end points of the cylinder
    dot_products_start = np.tensordot(img_array - start, vec, axes=([2], [0]))
    dot_products_end = np.tensordot(img_array - end, vec, axes=([2], [0]))
    # Generate a boolean mask indicating if a pixel lies within the cylinder
    mask = (dot_products_start >= 0) & (dot_products_end <= 0) & (np.linalg.norm(cross_products, axis=2) <= constraint)

    # Create the output image based on the mask, setting target color regions to white and all else to black
    output_array = np.where(mask[..., None], [255, 255, 255], [0, 0, 0])

    # Convert the numpy array back to an image for the final output
    output_image = Image.fromarray(np.uint8(output_array)).convert("RGB")

    return output_image


def text_from_greyscale(input_image_obj, COL):
    """
    Extracts and processes text from a greyscale image using OCR (Optical Character Recognition).
    
    This function applies preprocessing to the image to enhance text recognition, then uses
    tesseract OCR to extract text data. It groups the text into lines and words, filters out
    irrelevant parts of the image, and performs post-processing to structure the data into a
    DataFrame. It also includes matching of specific target words and extraction of associated
    numeric values and units, and uses known relationships between extracted metrics to correct
    errors in text recognition. The function utilizes PIL for image manipulation, numpy for array
    operations, scipy for image processing, pytesseract for OCR, and OpenCV for drawing bounding
    boxes around the text.

    Args:
        input_image_obj (str) : Name of file within current directory, or path to a file.
        COL (JpegImageFile) : PIL JpegImageFile of the filtered image highlighting yellow text.
    Returns:
        (tuple): tuple containing:
            - **Fail** (int) - Checks if the function has failed (1), or passed (0).
            - **df** (DataFrame) - Dataframe with columns 'Line', 'Word', 'Value', 'Unit'. populated with data extracted from the image with tesseract.

    """

    PIX = COL.load()
    img = input_image_obj

    # 2. Apply slight Gaussian blur
    # smoothed_image = COL.filter(ImageFilter.GaussianBlur(radius=1)) # In some cases smoothing helps, in others it makes it worse?

    for y in range(
            int(COL.size[1] * 0.45), COL.size[1]
    ):  # Exclude bottom 3rd of image - these are fails
        for x in range(COL.size[0]):
            PIX[x, y] = (0, 0, 0)

    pixels = COL  # np.array(smoothed_image)
    data = pytesseract.image_to_data(
        pixels, output_type=Output.DICT, lang="eng", config="--oem 1 --psm 3 -c tessedit_char_blacklist=l,!_|=$"
    )

    # This is rough, if more than 30 objects found then highly likely it is a waveform scan.
    Fail = 1 if len(data["text"]) < 30 else 0

    # Loop through each word and draw a box around it
    y_center = np.zeros(len(data["text"]))  # Variable to store the y-center of each bounding box of text detected.
    for i in range(len(data["text"])):
        if data["text"][i] != '' and data["text"][i] != ' ':
            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]
            if int(data["conf"][i]) > -1:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                y_center[i] = y + (h / 2)
            else:
                y_center[i] = 0
        else:
            y_center[i] = 0

    def group_similar_numbers(y_center, tolerance, OCR_data):
        # This function groups indexes of words with similar y-coordinate center
        # and also calculates the bounding box for each group of words
        words = OCR_data["text"]
        lefts = OCR_data["left"]
        tops = OCR_data["top"]
        widths = OCR_data["width"]
        heights = OCR_data["height"]
        x_centers = [left + width / 2 for left, width in zip(lefts, widths)]  # Calculate x_center for sorting

        # Convert y_center to a numpy array and reshape for DBSCAN
        data = np.array(y_center).reshape(-1, 1)

        # Perform DBSCAN clustering on y-coordinates
        dbscan = DBSCAN(eps=tolerance, min_samples=2)
        labels = dbscan.fit_predict(data)

        # Group the indices based on the cluster labels
        groups = {}
        for i, label in enumerate(labels):
            if label not in groups:
                groups[label] = []
            groups[label].append(i)

        # Group the words and calculate bounding boxes for each group
        grouped_words = []
        bounding_boxes = []
        for group_indices in groups.values():
            # Sort the indices within the group based on the 'x_center' value
            sorted_indices = sorted(group_indices, key=lambda idx: x_centers[idx])

            # Extract the sorted words based on the sorted indices
            group_words = [words[idx] for idx in sorted_indices]
            grouped_words.append(' '.join(group_words))

            # Calculate the bounding box for the current group
            group_lefts = [lefts[idx] for idx in sorted_indices]
            group_tops = [tops[idx] for idx in sorted_indices]
            group_rights = [lefts[idx] + widths[idx] for idx in sorted_indices]
            group_bottoms = [tops[idx] + heights[idx] for idx in sorted_indices]

            bounding_box = {
                'top_left': (min(group_lefts), min(group_tops)),
                'bottom_right': (max(group_rights), max(group_bottoms))
            }
            bounding_boxes.append(bounding_box)

        return grouped_words, bounding_boxes

    tolerance = 5  # Adjust the tolerance value - the max difference between y-coords considered on the same line
    grouped_words, bounding_boxes = group_similar_numbers(y_center, tolerance, data)

    # The bounding box is expected in the form of (left, upper, right, lower)
    bounding_box_index = 0 if len(bounding_boxes) == 1 else 1
    assert len(bounding_boxes) > 0

    left, top = bounding_boxes[bounding_box_index]['top_left']
    right, bottom = bounding_boxes[bounding_box_index]['bottom_right']
    crop_box = (left - 1, top - 1, right + 1, bottom - 1)
    cropped_image = COL.crop(crop_box)

    # def increase_dpi(image, factor=2):
    #     """Increases the DPI of an image by a factor

    #     Args:
    #         image (ndarray) : A 2D numpy array of the image.
    #         factor (int, optional) : The factor to increase the DPI by.

    #     Returns:
    #         col_image (ndarry) : A 2D numpy array of the image with increased DPI.
    #     """

    #     # Increases the number of rows
    #     row_image = np.zeros((image.shape[0] * factor, image.shape[1]))
    #     for i in range(image.shape[0]):
    #         new_rows = np.arange(i * factor, (i + 1) * factor)
    #         row_image[new_rows, :] = image[i, :]

    #     # Increases the number of cols
    #     col_image = np.zeros((row_image.shape[0], row_image.shape[1] * factor))
    #     for j in range(row_image.shape[1]):
    #         new_cols = np.arange(j * factor, (j + 1) * factor)
    #         col_image[:, new_cols] = row_image[:, j][..., None]

    #     return col_image

    # from skimage import filters, morphology

    # # Assuming 'increase_dpi' is a function you have defined to increase the DPI of the image
    # image_dpi = increase_dpi(np.array(cropped_image)[:,:,0], factor=4)

    # # Apply a Gaussian filter
    # image_gaussian = filters.gaussian(image_dpi, sigma=2)

    # image_final = image_gaussian
    # # # Perform the dilation
    # # # Create a structuring element, you can choose different shapes and sizes
    # # selem = morphology.disk(1)  # This creates a disk-shaped structuring element with radius 1

    # # # Apply the dilation
    # # image_dilated = morphology.dilation(image_gaussian, selem)

    # # # Convert the dilated image back to the original shape with the required number of channels
    # # image_final = np.zeros((image_dilated.shape[0], image_dilated.shape[1], np.array(cropped_image).shape[2]))

    # # # Add the dilated image back into each channel
    # # for i in range(np.array(cropped_image).shape[2]):
    # #     image_final[:, :, i] = image_dilated

    # # Convert to unsigned 8-bit integer type if necessary
    # image_final = image_final.astype(np.uint8)

    # # Performs binarisation of image
    # threshold = np.max(image_final) / 4
    # image_bin = np.zeros(image_final.shape)
    # image_bin[image_final < threshold] = 255
    # image_final = np.copy(image_bin.astype(np.uint8))

    # data2 = pytesseract.image_to_data(
    #     cropped_image, output_type=Output.DICT, lang="eng", config="--oem 1 --psm 7 -c tessedit_char_blacklist=l!~>»oOe¢_|=$"
    # )

    # # Loop through each word and draw a box around it
    # y_center = np.zeros(len(data2["text"])) # Variable to store the y-center of each bounding box of text detected.
    # for i in range(len(data2["text"])):
    #     if data2["text"][i] != '' and data2["text"][i] != ' ':
    #         x = data2["left"][i]
    #         y = data2["top"][i]
    #         w = data2["width"][i]
    #         h = data2["height"][i]
    #         if int(data2["conf"][i]) > -1:
    #             #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #             y_center[i] = y + (h/2)
    #         else:
    #             y_center[i] = 0  
    #     else:
    #         y_center[i] = 0

    # tolerance = 5  # Adjust the tolerance value - the max difference between y-coords considered on the same line
    # grouped_words, bounding_boxes = group_similar_numbers(y_center, 20, data2)
    # Display image
    plt.imshow(img)

    # Analyze the OCR output
    target_words = [
        "Lt Ut-PS",
        "Lt Ut-ED",
        "Lt Ut-S/D",
        "Lt Ut-PI",
        "Lt Ut-RI",
        "Lt Ut-MD",
        "Lt Ut-TAmax",
        "Lt Ut-HR",
        "Rt Ut-PS",
        "Rt Ut-ED",
        "Rt Ut-S/D",
        "Rt Ut-PI",
        "Rt Ut-RI",
        "Rt Ut-MD",
        "Rt Ut-TAmax",
        "Rt Ut-HR",
        "Umb-PS",
        "Umb-ED",
        "Umb-S/D",
        "Umb-PI",
        "Umb-RI",
        "Umb-MD",
        "Umb-TAmax",
        "Umb-HR",
        "DV-S",
        "DV-D",
        "DV-a",
        "DV-TAmax",
        "DV-S/a",
        "DV-a/S",
        "DV-PI",
        "DV-PLI",
        "DV-PVIV",
        "DV-HR",
    ]

    # Split text into lines
    lines = grouped_words  # text.split("\n")
    # Initialize DataFrame
    df = pd.DataFrame(columns=["Line", "Word", "Value", "Unit"])

    prefixes = ["Lt", "Rt", "Umb", "DV"]
    prefix_counts = {prefix: sum(1 for line in lines if prefix in line) for prefix in prefixes}
    most_likely_prefix = max(prefix_counts, key=prefix_counts.get)

    # Filter target words based on the most likely prefix
    target_words = [word for word in target_words if word.startswith(most_likely_prefix)]
    word_order = [word for word in target_words if word.startswith(most_likely_prefix)]
    target_word_mem = target_words.copy()
    # Step 1: Exact matching
    matched_lines = set()  # to store the indices of lines that have been matched

    for i, line in enumerate(lines):
        for word in target_words:
            if word in line:  # checking for exact match
                # Extract value and unit
                match = re.search(r"(\-?\d+(\s*\d+)*\.\s*\d+|\-?\d+(\s*\d+)*)\s*([^\d\s]+)?$", line)

                if match:
                    value = float(match.group(1).replace(' ', ''))
                    unit = match.group(4) if match.group(4) else ""
                    df.loc[len(df)] = {"Line": i + 1, "Word": word, "Value": value, "Unit": unit}
                    target_words.remove(word)
                else:
                    # logger.warning("couldn't find numeric data for line.")
                    df.loc[len(df)] = {"Line": i + 1, "Word": word, "Value": 0, "Unit": 0}
                    target_words.remove(word)
                matched_lines.add(i)
                break  # Exit the inner loop once a match is found

    def is_subsequence(target, line):
        target_idx = 0
        line_idx = 0

        # Filter out spaces and hyphens from target
        filtered_target = [char for char in target if char not in [' ', '-']]

        while target_idx < len(filtered_target) and line_idx < len(line):
            if filtered_target[target_idx].lower() == line[line_idx].lower():
                target_idx += 1
            line_idx += 1

        return target_idx == len(filtered_target)

    # Step 2: Subsequence matching for unmatched lines
    for i, line in enumerate(lines):
        if i not in matched_lines:  # only process unmatched lines
            for word in target_words:
                if is_subsequence(word, line):
                    # Extract value and unit
                    match = re.search(r"(\-?\d+\.\d+|\-?\d+)\s*([^\d\s]+)?$", line)
                    if match:
                        value = float(match.group(1))
                        unit = match.group(2) if match.group(2) else ""
                        df.loc[len(df)] = {"Line": i + 1, "Word": word, "Value": value, "Unit": unit}
                        target_words.remove(word)
                    else:
                        # logger.warning("couldn't find numeric data for line.")
                        df.loc[len(df)] = {"Line": i + 1, "Word": word, "Value": 0, "Unit": 0}
                        target_words.remove(word)
                    matched_lines.add(i)
                    break  # Exit the inner loop once a match is found

    def find_closest_target(line, target_words):
        min_distance = float('inf')
        closest_word = None

        for word in target_words:
            distance = Levenshtein.distance(line, word)
            if distance < min_distance:
                min_distance = distance
                closest_word = word

        return closest_word, min_distance

    # Set a threshold for acceptable similarity
    threshold = 7

    for i, line in enumerate(lines):
        if i not in matched_lines:  # only process unmatched lines
            closest_word, distance = find_closest_target(line, target_words)

            if distance <= threshold:
                # Extract value and unit
                match = re.search(r"(\-?\d+(\s*\d+)*\.\s*\d+|\-?\d+(\s*\d+)*)\s*([^\d\s]+)?$", line)
                if match:
                    value = float(match.group(1).replace(' ', ''))
                    unit = match.group(4) if match.group(4) else ""
                    df.loc[len(df)] = {"Line": i + 1, "Word": closest_word, "Value": value, "Unit": unit}
                    target_words.remove(closest_word)
                matched_lines.add(i)

    target_words_extended = [
        "Lt Ut-PS cm/s",
        "Lt Ut-ED cm/s",
        "Lt Ut-S/D",
        "Lt Ut-PI",
        "Lt Ut-RI",
        "Lt Ut-MD cm/s",
        "Lt Ut-TAmax cm/s",
        "Lt Ut-HR bpm",
        "Rt Ut-PS cm/s",
        "Rt Ut-ED cm/s",
        "Rt Ut-S/D",
        "Rt Ut-PI",
        "Rt Ut-RI",
        "Rt Ut-MD cm/s",
        "Rt Ut-TAmax cm/s",
        "Rt Ut-HR bpm",
        "Umb-PS cm/s",
        "Umb-ED cm/s",
        "Umb-S/D",
        "Umb-PI",
        "Umb-RI",
        "Umb-MD cm/s",
        "Umb-TAmax cm/s",
        "Umb-HR bpm",
        "DV-S cm/s",
        "DV-D cm/s",
        "DV-a",
        "DV-TAmax cm/s",
        "DV-S/a",
        "DV-a/S",
        "DV-PI",
        "DV-PLI",
        "DV-PVIV",
        "DV-HR bpm",
    ]

    if target_words:

        suffixes = [word.split('-')[-1] for word in target_words if '-' in word]
        indices = [i for i, entry in enumerate(target_words_extended) if any(sub in entry for sub in target_words)]
        remaining_target_extended = [target_words_extended[i] for i in indices]

        # Create a distance matrix
        num_lines = len(lines)
        num_target_words = len(remaining_target_extended)
        distance_matrix = np.zeros((num_lines, num_target_words))

        # Bias values - these need to be defined by you, for example:
        # Create a list of bias values, all set to -2, with the same length as suffixes
        bias_values = [-2] * len(suffixes)  # Creates a list with -2 repeated len(suffixes) times

        # Bias dictionary, where the keys are suffixes and the values are the bias amounts
        bias_dict = {suffix: bias for suffix, bias in zip(suffixes, bias_values)}

        # Initialize your distance matrix
        num_lines = len(lines)
        num_target_words = len(remaining_target_extended)
        distance_matrix = np.zeros((num_lines, num_target_words))

        # Calculate biased distances
        for i, line in enumerate(lines):
            if i not in matched_lines:
                for j, word in enumerate(remaining_target_extended):
                    # Remove digits from the line
                    line_no_digits = re.sub(r'\d+', '', line)

                    # Calculate basic Levenshtein distance
                    basic_distance = Levenshtein.distance(line_no_digits, word)

                    # Apply bias if a specific suffix is expected in the line
                    expected_suffix = suffixes[j]  # Suffix that corresponds to the current target word
                    if expected_suffix in line:
                        # Subtract bias to reduce distance
                        basic_distance += bias_dict[expected_suffix]

                    # Set the biased distance in the matrix
                    distance_matrix[i, j] = basic_distance

        matches = {}
        for j, word in enumerate(remaining_target_extended):
            # Find the line with the smallest non-zero distance for the current target word
            line_indices_with_non_zero_distances = np.where(distance_matrix[:, j] > 0)[0]
            if len(line_indices_with_non_zero_distances) > 0:
                i = line_indices_with_non_zero_distances[np.argmin(distance_matrix[line_indices_with_non_zero_distances, j])]
                line = lines[i]
                # Add to matches
                matches[line] = word

                # Extract value and unit from the line and add to the DataFrame
                match = re.search(r"(\-?\d+(\s*\d+)*\.\s*\d+|\-?\d+(\s*\d+)*)\s*([^\d\s]+)?$", line)
                if match:
                    value = float(match.group(1).replace(' ', ''))
                    unit = match.group(4) if match.group(4) else ""
                    df.loc[len(df)] = {"Line": i + 1, "Word": target_words[j], "Value": value, "Unit": unit}
                else:
                    df.loc[len(df)] = {"Line": i + 1, "Word": target_words[j], "Value": 0, "Unit": 0}
                matched_lines.add(i)

                # Remove the matched line from further consideration
                distance_matrix[i, :] = np.inf

    # Create a mask for each word in the word_order list and concatenate them in order
    df = pd.concat([df.loc[df['Word'] == word] for word in word_order]).reset_index(drop=True)

    try:  # This is still a test really
        if most_likely_prefix == "DV":

            if df.loc[df['Word'] == 'DV-D', 'Value'].values[0] > df.loc[df['Word'] == 'DV-S', 'Value'].values[0] and df.loc[df['Word'] == 'DV-S/a', 'Value'].values[0] > \
                    df.loc[df['Word'] == 'DV-S', 'Value'].values[0] and df.loc[df['Word'] == 'DV-S/a', 'Unit'].values[0] != '':
                # Storing temporary values for swapping
                temp = df.loc[df['Word'] == 'DV-S/a', 'Value'].values[0]
                df.loc[df['Word'] == 'DV-S/a', 'Value'] = df.loc[df['Word'] == 'DV-S', 'Value'].values[0]
                df.loc[df['Word'] == 'DV-S', 'Value'] = temp
                print("swapped DV-S/a and DV-S")

            if df.loc[df['Word'] == 'DV-a/S', 'Unit'].values[0] != '' and df.loc[df['Word'] == 'DV-a', 'Unit'].values[0] == '':
                # Storing temporary values for swapping
                temp = df.loc[df['Word'] == 'DV-a/S', 'Value'].values[0]
                temp_unit = df.loc[df['Word'] == 'DV-a/S', 'Unit'].values[0]
                df.loc[df['Word'] == 'DV-a/S', 'Value'] = df.loc[df['Word'] == 'DV-a', 'Value'].values[0]
                df.loc[df['Word'] == 'DV-a/S', 'Unit'] = df.loc[df['Word'] == 'DV-a', 'Unit'].values[0]
                df.loc[df['Word'] == 'DV-a', 'Value'] = temp
                df.loc[df['Word'] == 'DV-a', 'Unit'] = temp_unit

            df = metric_check_dv(df)  # handle the ductus venousus differently
        else:
            df = metric_check(df)  # for left, right, and umbilical
    except:
        print("metric check failed")

    # Enforce positive heart-rate values: OCR occasionally hallucinates a leading '-'
    if not df.empty and 'Value' in df.columns and 'Word' in df.columns:
        hr_mask = df['Word'].str.contains('HR', na=False) & df['Value'].notna()
        df.loc[hr_mask, 'Value'] = df.loc[hr_mask, 'Value'].abs()

    return Fail, df


def metric_check(df):
    """Performs validation and correction of ultrasound measurement metrics within a DataFrame.

    This function identifies the prefix used in the metrics (either left, right, or uterine artery),
    checks for missing rows,and adds them if necessary. It applies common sense checks to the pulsatility index (PI) 
    and time-averaged maximum velocity (TAmax) values to correct common OCR errors. The function 
    also adjusts the peak systolic (PS) and end diastolic (ED) velocity values based on their 
    relationship with other metrics, ensuring consistency. Lastly, it calculates the systolic
    over diastolic ratio (S/D), resistance index (RI), and TAmax from the corrected PS and ED 
    values and checks them against the extracted metrics for consistency.

    Args:
        - **df** (DataFrame): Extracted data from image

    Returns:
        - df (DataFrame): DataFrame with corrected values after metric checking calculations
    """

    def identify_prefix(lines):
        # Try to identify the prefix in use
        for prefix in ["Lt", "Rt", "Umb"]:
            if lines['Word'].str.contains(prefix).any():
                print("prefix found")
                PRF = prefix

            target_words = [
                "Lt Ut-PS",
                "Lt Ut-ED",
                "Lt Ut-S/D",
                "Lt Ut-PI",
                "Lt Ut-RI",
                "Lt Ut-MD",
                "Lt Ut-TAmax",
                "Lt Ut-HR",
                "Rt Ut-PS",
                "Rt Ut-ED",
                "Rt Ut-S/D",
                "Rt Ut-PI",
                "Rt Ut-RI",
                "Rt Ut-MD",
                "Rt Ut-TAmax",
                "Rt Ut-HR",
                "Umb-PS",
                "Umb-ED",
                "Umb-S/D",
                "Umb-PI",
                "Umb-RI",
                "Umb-MD",
                "Umb-TAmax",
                "Umb-HR",
            ]
        # Splitting the target words based on prefixes
        target_words = [word for word in target_words if word.startswith(PRF)]

        return PRF, target_words  # Return None if no known prefix is found

    def add_missing_rows(df_in):
        # Identify the Prefix
        prefix, target_words = identify_prefix(df_in)

        # Determine Missing Rows
        existing_words = df_in['Word'].tolist()
        missing_targets = [word for word in target_words if word not in existing_words]

        # Add Missing Rows
        for target in missing_targets:
            df_in.loc[len(df_in)] = {"Word": target, "Value": 0, "Unit": ""}

        return df_in

    #df = add_missing_rows(df)

    def check_TAmax_value(value_in, df_in):  # Decimal can be misread, so common sense check.

        MD = df_in.loc[df['Word'].str.contains('MD'), 'Value'].values[0] if df_in['Word'].str.contains('MD').any() else 0
        PS = df_in.loc[df['Word'].str.contains('PS'), 'Value'].values[0] if df_in['Word'].str.contains('PS').any() else 0
        ED = df_in.loc[df['Word'].str.contains('ED'), 'Value'].values[0] if df_in['Word'].str.contains('ED').any() else 0

        # If the other values are positive, return the absolute value of TAmax
        if value_in < 0 < MD and PS > 0 and ED > 0:
            return abs(value_in)

        return value_in  # or return some default value or raise an exception

    # Sense check some values:
    PI = df.loc[df['Word'].str.contains('PI'), 'Value'].values[0] if df['Word'].str.contains('PI').any() else 0
    df.loc[df['Word'].str.contains('PI'), 'Value'] = check_pi_value(PI)
    TAmax = df.loc[df['Word'].str.contains('TAmax'), 'Value'].values[0] if df['Word'].str.contains('TAmax').any() else 0
    df.loc[df['Word'].str.contains('TAmax'), 'Value'] = check_TAmax_value(TAmax, df)

    # Peak systolic
    PS = df.loc[df['Word'].str.contains('PS'), 'Value'].values[0] if df['Word'].str.contains('PS').any() else 0
    # End diastolic
    ED = df.loc[df['Word'].str.contains('ED'), 'Value'].values[0] if df['Word'].str.contains('ED').any() else 0

    def check_TAmax_value(PS, ED, df):  # Decimal can be misread, so common sense check.

        TAmax = df.loc[df['Word'].str.contains('TAmax'), 'Value'].values[0] if df['Word'].str.contains('TAmax').any() else 0
        MD = df.loc[df['Word'].str.contains('MD'), 'Value'].values[0] if df['Word'].str.contains('MD').any() else 0

        # Check if there's a difference in sign between PS and ED
        if (PS > 0 > ED) or (PS < 0 < ED):
            # Check if TAmax and MD have the same sign
            if (TAmax > 0 and MD > 0) or (TAmax < 0 and MD < 0):
                # If so, change the sign of PS and ED to match that of TAmax and MD
                PS = abs(PS) if TAmax > 0 else -abs(PS)
                ED = abs(ED) if TAmax > 0 else -abs(ED)

        PSnew = PS
        EDnew = ED
        # If the value is between 3 and 10, divide it by 10
        if 250 <= PS <= 1000:
            PSnew = PS / 10

        # If the value is between 10 and 200, divide it by 100
        if 1000 <= PS <= 10000:
            PSnew = PS / 100

        # If the value is between 3 and 10, divide it by 10
        if 200 <= ED <= 1000:
            EDnew = ED / 10

        # If the value is between 10 and 200, divide it by 100
        if 1000 <= ED <= 10000:
            EDnew = ED / 100

        return PSnew, EDnew

    PS, ED = check_TAmax_value(PS, ED, df)  # sense check for pressures
    df.loc[df['Word'].str.contains('PS'), 'Value'] = PS
    df.loc[df['Word'].str.contains('ED'), 'Value'] = ED

    # Find S/D
    SoverD_calc = PS / ED
    # Find RI
    RI_calc = (PS - ED) / PS
    # Find TAmax
    TAmax_calc = (PS + (2 * ED)) / 3

    # Now check whether the PS & ED dependant metrics are consistent between calculated and extracted:
    # Extracted values with default as None if not present
    SoverD_extracted = df.loc[df['Word'].str.contains('S/D'), 'Value'].values[0] if df['Word'].str.contains('S/D').any() else None
    RI_extracted = df.loc[df['Word'].str.contains('RI'), 'Value'].values[0] if df['Word'].str.contains('RI').any() else None
    TAmax_extracted = df.loc[df['Word'].str.contains('TAmax'), 'Value'].values[0] if df['Word'].str.contains('TAmax').any() else None
    comparison_dataframe = pd.DataFrame(index=['PS_extracted', 'ED_extracted', 'SoverD_extracted', 'RI_extracted', 'TAmax_extracted',
                                               'SoverD_calc', 'RI_calc', 'TAmax_calc', 'PS_calc', 'ED_calc', 'ED_from_SoverD', 'ED_from_RI',
                                               'SoverD_from_ED_from_RI', 'RI_from_ED_from_SoverD', 'TAmax_from_ED_from_SoverD', 'TAmax_from_ED_from_RI',
                                               'PS_from_SoverD', 'PS_from_RI', 'SoverD_from_PS_from_RI', 'RI_from_PS_from_SoverD', 'TAmax_from_PS_from_SoverD',
                                               'TAmax_from_PS_from_RI'], columns=['Extracted'])
    # List of all the extracted metrics that exist
    existing_metrics = [metric for metric in [SoverD_extracted, RI_extracted, TAmax_extracted] if metric is not None]
    values_to_insert = {
        'PS_extracted': PS,
        'ED_extracted': ED,
        'SoverD_extracted': SoverD_extracted,
        'RI_extracted': RI_extracted,
        'TAmax_extracted': TAmax_extracted
    }

    for key, value in values_to_insert.items():
        comparison_dataframe.loc[key, 'Extracted'] = value
        # Check closeness and store conditions met in a list

    values_to_insert = {
        'SoverD_calc': SoverD_calc,
        'RI_calc': RI_calc,
        'TAmax_calc': TAmax_calc
    }
    for key, value in values_to_insert.items():
        comparison_dataframe.loc[key, 'First_calc'] = value
        # Check closeness and store conditions met in a list

    def Metric_comparison(c_df, col):

        # Tolerance level (you can adjust this based on your requirements)
        tolerance1 = 0.2
        tolerance2 = 4  # This tolerance is larger because the equation we used for TAmax is approximate
        conditions_met = []
        for parameter, extracted_name in [('SoverD', 'SoverD_extracted'), ('RI', 'RI_extracted'), ('TAmax', 'TAmax_extracted')]:
            extracted_value = c_df['Extracted'][extracted_name]

            if extracted_value is not None:
                for row_name, calc_value in c_df.iloc[:, col].items():
                    if str(row_name).startswith(extracted_name[:-9]):  # If row name starts with the parameter name
                        tolerance = tolerance1 if parameter != 'TAmax' else tolerance2
                        if abs(calc_value - extracted_value) < tolerance:
                            conditions_met.append(row_name)
                            break  # Exit the inner loop once a match is found

        return conditions_met

    conditions_met = Metric_comparison(comparison_dataframe, 1)

    # You've already extracted PS, SoverD_extracted, RI_extracted, and TAmax_extracted

    # Check if all 3 metrics are inconsistent
    if len(conditions_met) == 0:  # All 3 are not consistent
        # Assume PS was extracted correctly, compute ED from the 3 metrics:
        try:
            # Recalculate ED using extracted metrics and assumed correct PS
            ED_from_SoverD = PS / SoverD_extracted if SoverD_extracted else None
            ED_from_RI = PS * (1 - RI_extracted) if RI_extracted else None
            # Now, using these new ED values, recalculate the metrics
            SoverD_from_ED_from_RI = PS / ED_from_RI if ED_from_RI else None
            RI_from_ED_from_SoverD = (PS - ED_from_SoverD) / PS if ED_from_SoverD else None
            TAmax_from_ED_from_SoverD = (PS + 2 * ED_from_SoverD) / 3 if ED_from_SoverD else None
            TAmax_from_ED_from_RI = (PS + 2 * ED_from_RI) / 3 if ED_from_RI else None

            values_to_insert = {
                'ED_from_SoverD': ED_from_SoverD,
                'ED_from_RI': ED_from_RI,
                'SoverD_from_ED_from_RI': SoverD_from_ED_from_RI,
                'RI_from_ED_from_SoverD': RI_from_ED_from_SoverD,
                'TAmax_from_ED_from_SoverD': TAmax_from_ED_from_SoverD,
                'TAmax_from_ED_from_RI': TAmax_from_ED_from_RI
            }
            for key, value in values_to_insert.items():
                comparison_dataframe.loc[key, 'Second_calc'] = value
                # Check closeness and store conditions met in a list

            # Check conditions met with these values:
            conditions_met = Metric_comparison(comparison_dataframe, 2)

            if len(conditions_met) == 0 or (ED_from_RI < 2 and ED_from_SoverD < 2):  # If our assumption above was wrong
                # Recalculate PS using the extracted metrics and assumed correct ED
                PS_from_SoverD = SoverD_extracted * ED if SoverD_extracted else None
                PS_from_RI = ED / (1 - RI_extracted) if RI_extracted else None
                # Now, using these new PS values, recalculate the other metrics
                SoverD_from_PS_from_RI = PS_from_RI / ED if PS_from_RI else None
                RI_from_PS_from_SoverD = (PS_from_SoverD - ED) / PS_from_SoverD if PS_from_SoverD else None
                TAmax_from_PS_from_SoverD = (PS_from_SoverD + 2 * ED) / 3 if PS_from_SoverD else None
                TAmax_from_PS_from_RI = (PS_from_RI + 2 * ED) / 3 if PS_from_RI else None

                values_to_insert = {
                    'PS_from_SoverD': PS_from_SoverD,
                    'PS_from_RI': PS_from_RI,
                    'SoverD_from_PS_from_RI': SoverD_from_PS_from_RI,
                    'RI_from_PS_from_SoverD': RI_from_PS_from_SoverD,
                    'TAmax_from_PS_from_SoverD': TAmax_from_PS_from_SoverD,
                    'TAmax_from_PS_from_RI': TAmax_from_PS_from_RI
                }
                for key, value in values_to_insert.items():
                    comparison_dataframe.loc[key, 'Third_calc'] = value
                    # Check closeness and store conditions met in a list

                conditions_met = Metric_comparison(comparison_dataframe, 3)
                # now check the conditions again:
                if len(conditions_met) > 0:
                    row_name = conditions_met[0]

                    parts = row_name.split('_from_')
                    desired_row_name = parts[1] + '_from_' + parts[2]
                    new_value = comparison_dataframe.loc[desired_row_name, 'Third_calc']
                    # We have calculated the new PS!
                    print(f"Recalculated PS (from RI): {new_value}")
                    df.loc[df['Word'].str.contains('PS'), 'Value'] = round(new_value, 2)
                    PS = df.loc[df['Word'].str.contains('PS'), 'Value'].values[0]
                    ED = df.loc[df['Word'].str.contains('ED'), 'Value'].values[0]
                    # Find S/D
                    df.loc[df['Word'].str.contains('S/D'), 'Value'] = round(PS / ED, 2)
                    # Find RI
                    df.loc[df['Word'].str.contains('RI'), 'Value'] = round((PS - ED) / PS, 2)
                    # Find TAmax
                    df.loc[df['Word'].str.contains('TAmax'), 'Value'] = round((PS + (2 * ED)) / 3, 2)

            elif len(conditions_met) > 0:

                row_name = conditions_met[0]

                parts = row_name.split('_from_')
                desired_row_name = parts[1] + '_from_' + parts[2]
                new_value = comparison_dataframe.loc[desired_row_name, 'Second_calc']
                # We have calculated the new PS!
                print(f"Recalculated PS (from RI): {new_value}")
                df.loc[df['Word'].str.contains('ED'), 'Value'] = round(new_value, 2)
                PS = df.loc[df['Word'].str.contains('PS'), 'Value'].values[0]
                ED = df.loc[df['Word'].str.contains('ED'), 'Value'].values[0]
                # Find S/D
                df.loc[df['Word'].str.contains('S/D'), 'Value'] = round(PS / ED, 2)
                # Find RI
                df.loc[df['Word'].str.contains('RI'), 'Value'] = round((PS - ED) / PS, 2)
                # Find TAmax
                df.loc[df['Word'].str.contains('TAmax'), 'Value'] = round((PS + (2 * ED)) / 3, 2)


        except ZeroDivisionError:
            print("Error: Division by zero encountered. Check the extracted values.")
    elif len(conditions_met) < 3:
        # At least 1 of the metrics is consistent, therefore PS and ED can be assumed to be correct,
        # Caclulate the inconsistent metrics from the PS and ED calculations
        print("At least one text extraction error, correcting...")

        if 'SoverD' not in conditions_met:
            # Find S/D
            df.loc[df['Word'].str.contains('S/D'), 'Value'] = round(PS / ED, 2)
        if 'RI' not in conditions_met:
            # Find RI
            df.loc[df['Word'].str.contains('RI'), 'Value'] = round((PS - ED) / PS, 2)
        if 'TAmax' not in conditions_met:
            # Find TAmax
            df.loc[df['Word'].str.contains('TAmax'), 'Value'] = round((PS + (2 * ED)) / 3, 2)
    else:
        print("All metrics are consistent.")

    return df


def upscale_both_images(PIL_img, cv2_img, max_length=950, min_length=950):
    """ **For testing improved text extraction**

    Up-scales both a PIL and an OpenCV image to a specified maximum length while
    maintaining their aspect ratios. If the longest edge of an image is already 
    greater than or equal to the specified minimum length, the image will not be up-scaled.

    Args:
        PIL_img (PIL.Image.Image): The image to upscale using the PIL library.
        cv2_img (numpy.ndarray): The image to upscale using the OpenCV library.
        max_length (int, optional): The maximum length of the longest edge after up-scaling.
                                    Defaults to 950.
        min_length (int, optional): The minimum length required to trigger up-scaling.
                                    If the longest edge of the image is already greater 
                                    than or equal to this length, up-scaling does not occur.
                                    Defaults to 950.

    Returns:
        tuple: A tuple containing the up-scaled PIL.Image.Image and up-scaled numpy.ndarray (OpenCV image) respectively.
    """

    def upscale_image(image, is_pil):
        # Get original dimensions
        if is_pil:
            original_width, original_height = image.size
        else:  # OpenCV image
            original_height, original_width = image.shape[:2]

        # Check if the longest edge is already greater than or equal to min_length
        if max(original_width, original_height) >= min_length:
            return image

        # Determine the scaling factor
        scaling_factor = max_length / max(original_width, original_height)

        # Calculate the new size
        new_width = int(original_width * scaling_factor)
        new_height = int(original_height * scaling_factor)

        # Resize the image
        if is_pil:
            return image.resize((new_width, new_height), Image.LANCZOS)
        else:  # OpenCV image
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Upscale both images
    upscaled_PIL_img = upscale_image(PIL_img, is_pil=True)
    upscaled_cv2_img = upscale_image(cv2_img, is_pil=False)

    return upscaled_PIL_img, upscaled_cv2_img


def check_pi_value(value):  # Decimal can be misread, so common sense check.
    # If the value is between 0 and 2, return it as is
    if 0 <= value <= 3:
        return value

    # If the value is between 3 and 10, divide it by 10
    if 3 <= value <= 10:
        return value / 10

    # If the value is between 10 and 200, divide it by 100
    if 10 <= value <= 200:
        return value / 100

    # If the value is outside of these ranges, return a default or handle accordingly
    return value  # or return some default value or raise an exception


def metric_check_dv(df):
    """
    Performs validation and correction of ultrasound measurement metrics within a DataFrame
    for Ductus Venosus (DV).

    This function performs the same function as Metric_check, but for Ductus Venosus. Metric_check
    works for left, right, and umbilical arteries as their scans all contain the same patient measurements,
    but DV scans have a separate set of measurements with their own unique relationship - this function
    corrects the values for DV scans.

    Args:
        df (DataFrame): Extracted data from image

    Returns:
        df(DataFrame): DataFrame with corrected values after metric checking calculations
    """

    # Splitting the target words based on prefixes
    def add_missing_rows(df):
        # Identify the Prefix
        prefix = "DV"

        target_words = ["DV-S",
                        "DV-D",
                        "DV-a",
                        "DV-TAmax",
                        "DV-S/a",
                        "DV-a/S",
                        "DV-PI",
                        "DV-PLI",
                        "DV-PVIV",
                        "DV-HR", ]
        # Determine Missing Rows
        existing_words = df['Word'].tolist()
        missing_targets = [word for word in target_words if word not in existing_words]

        # Add Missing Rows
        for target in missing_targets:
            df.loc[len(df)] = {"Word": target, "Value": 0, "Unit": ""}

        return df

    df = add_missing_rows(df)

    def check_TAmax_value(value, df):  # Decimal can be misread, so common sense check.

        PLI = df.loc[df['Word'] == 'DV-S/a', 'Value'].values[0]
        PS = df.loc[df['Word'] == 'DV-S', 'Value'].values[0]
        ED = df.loc[df['Word'] == 'DV-D', 'Value'].values[0]

        # If the other values are positive, return the absolute value of TAmax
        if value < 0 < PLI and PS > 0 and ED > 0:
            return abs(value)

        return value  # or return some default value or raise an exception

    # Sense check some values:
    PI = df.loc[df['Word'] == 'DV-PI', 'Value'].values[0]
    df.loc[df['Word'] == 'DV-PI', 'Value'] = check_pi_value(PI)
    TAmax = df.loc[df['Word'] == 'DV-TAmax', 'Value'].values[0]
    df.loc[df['Word'] == 'DV-TAmax', 'Value'] = check_TAmax_value(TAmax, df)

    # Peak systolic
    PS = df.loc[df['Word'] == 'DV-S', 'Value'].values[0]
    # End diastolic
    ED = df.loc[df['Word'] == 'DV-D', 'Value'].values[0]

    def check_TAmax_value(PS, ED, df):  # Decimal can be misread, so common sense check.

        TAmax = df.loc[df['Word'] == 'DV-TAmax', 'Value'].values[0]
        a = df.loc[df['Word'] == 'DV-a', 'Value'].values[0]

        # Check if there's a difference in sign between PS and ED
        if (PS > 0 and ED < 0) or (PS < 0 and ED > 0):
            # Check if TAmax and MD have the same sign
            if (TAmax > 0 and a > 0) or (TAmax < 0 and a < 0):
                # If so, change the sign of PS and ED to match that of TAmax and MD
                PS = abs(PS) if TAmax > 0 else -abs(PS)
                ED = abs(ED) if TAmax > 0 else -abs(ED)

        return PS, ED

    PS, ED = check_TAmax_value(PS, ED, df)  # sense check for pressures

    def check_S_D_value(value):  # Decimal can be misread, so common sense check.
        # If the value is between 0 and 2, return it as is
        if 0 <= abs(value) <= 60:
            return value

        # If the value is between 3 and 10, divide it by 10
        if 60 <= abs(value) <= 600:
            return value / 10

        # If the value is between 10 and 200, divide it by 100
        if 600 <= abs(value):
            return value / 100

        # If the value is outside of these ranges, return a default or handle accordingly
        return value  # or return some default value or raise an exception

    ED = check_S_D_value(ED)
    df.loc[df['Word'] == 'DV-S', 'Value'] = PS
    df.loc[df['Word'] == 'DV-D', 'Value'] = ED
    a = df.loc[df['Word'] == 'DV-a', 'Value'].values[0]

    # Find S/D
    Sovera_calc = PS / a
    # Find RI
    aoverS_calc = a / PS
    # Find TAmax
    TAmax_calc = (PS + (2 * a)) / 3
    # Find PI
    PI_calc = (PS - a) / ((PS + a) / 2)

    # Now check whether the PS & a dependant metrics are consistent between calculated and extracted:
    # Extracted values with default as None if not present
    Sovera_extracted = df.loc[df['Word'] == 'DV-S/a', 'Value'].values[0]
    aoverS_extracted = df.loc[df['Word'] == 'DV-a/S', 'Value'].values[0]
    TAmax_extracted = df.loc[df['Word'] == 'DV-TAmax', 'Value'].values[0]
    PI_extracted = df.loc[df['Word'] == 'DV-PI', 'Value'].values[0]
    comparison_dataframe = pd.DataFrame(index=['PS_extracted', 'a_extracted', 'Sovera_extracted', 'TAmax_extracted', 'PI_extracted',
                                               'Sovera_calc', 'PI_calc', 'TAmax_calc', 'PS_calc', 'a_calc', 'a_from_Sovera', 'a_from_PI',
                                               'Sovera_from_a_from_PI', 'PI_from_a_from_Sovera', 'TAmax_from_a_from_Sovera', 'TAmax_from_a_from_PI',
                                               'PS_from_Sovera', 'PS_from_PI', 'Sovera_from_PS_from_PI', 'PI_from_PS_from_Sovera', 'TAmax_from_PS_from_Sovera',
                                               'TAmax_from_PS_from_PI'], columns=['Extracted'])
    # List of all the extracted metrics that exist
    existing_metrics = [metric for metric in [Sovera_extracted, PI_extracted, TAmax_extracted] if metric is not None]
    values_to_insert = {
        'PS_extracted': PS,
        'a_extracted': a,
        'Sovera_extracted': Sovera_extracted,
        'PI_extracted': PI_extracted,
        'TAmax_extracted': TAmax_extracted
    }

    for key, value in values_to_insert.items():
        comparison_dataframe.loc[key, 'Extracted'] = value
        # Check closeness and store conditions met in a list

    values_to_insert = {
        'Sovera_calc': Sovera_calc,
        'PI_calc': PI_calc,
        'TAmax_calc': TAmax_calc
    }
    for key, value in values_to_insert.items():
        comparison_dataframe.loc[key, 'First_calc'] = value
        # Check closeness and store conditions met in a list

    def metric_comparison(c_df, col):

        # Tolerance level (you can adjust this based on your requirements)
        tolerance1 = 0.2
        tolerance2 = 2  # This tolerance is larger because the equation we used for TAmax is approximate
        local_conditions_met = []
        for parameter, extracted_name in [('Sovera', 'Sovera_extracted'), ('PI', 'PI_extracted'), ('TAmax', 'TAmax_extracted')]:
            extracted_value = c_df['Extracted'][extracted_name]

            if extracted_value is not None:
                for local_row_name, calc_value in c_df.iloc[:, col].items():
                    if str(local_row_name).startswith(extracted_name[:-9]) and np.isnan(calc_value) != True:  # If row name starts with the parameter name
                        tolerance = tolerance1 if parameter != 'TAmax' else tolerance2
                        if abs(calc_value - extracted_value) < tolerance:
                            local_conditions_met.append(local_row_name)
                            break  # Exit the inner loop once a match is found

        return local_conditions_met

    try:
        conditions_met = metric_comparison(comparison_dataframe, 1)
        print("ok")
    except:
        conditions_met = None
        traceback.print_exc()

    # You've already extracted PS, Sovera_extracted, RI_extracted, and TAmax_extracted

    # Check if all 3 metrics are inconsistent
    if len(conditions_met) == 0:  # All 3 are not consistent
        # Assume PS was extracted correctly, compute a from the 3 metrics:
        try:
            # Recalculate a using extracted metrics and assumed correct PS
            a_from_Sovera = PS / Sovera_extracted if Sovera_extracted else None
            a_from_PI = PS * (1 - PI_extracted) if PI_extracted else None
            # Now, using these new a values, recalculate the metrics
            Sovera_from_a_from_PI = PS / a_from_PI if a_from_PI else None
            PI_from_a_from_Sovera = (PS - a_from_Sovera) / ((PS + a_from_Sovera) / 2) if a_from_Sovera else None
            TAmax_from_a_from_Sovera = (PS + (2 * a_from_Sovera)) / 3 if a_from_Sovera else None
            TAmax_from_a_from_PI = (PS + (2 * a_from_PI)) / 3 if a_from_PI else None

            values_to_insert = {
                'a_from_Sovera': a_from_Sovera,
                'a_from_PI': a_from_PI,
                'Sovera_from_a_from_PI': Sovera_from_a_from_PI,
                'PI_from_a_from_Sovera': PI_from_a_from_Sovera,
                'TAmax_from_a_from_Sovera': TAmax_from_a_from_Sovera,
                'TAmax_from_a_from_PI': TAmax_from_a_from_PI
            }
            for key, value in values_to_insert.items():
                comparison_dataframe.loc[key, 'Second_calc'] = value
                # Check closeness and store conditions met in a list

            # Check conditions met with these values:
            conditions_met = metric_comparison(comparison_dataframe, 2)
            if len(conditions_met) == 0:  # If our assumption above was wrong
                # Recalculate PS using the extracted metrics and assumed correct a
                PS_from_Sovera = Sovera_extracted * a if Sovera_extracted else None
                PS_from_PI = ((3 * PI_extracted) - (2 * a)) if PI_extracted else None
                # Now, using these new PS values, recalculate the other metrics
                Sovera_from_PS_from_PI = PS_from_PI / a if PS_from_PI else None
                PI_from_PS_from_Sovera = (PS_from_Sovera - a) / ((PS_from_Sovera + a) / 2) if PS_from_Sovera else None
                TAmax_from_PS_from_Sovera = (PS_from_Sovera + (2 * a)) / 3 if PS_from_Sovera else None
                TAmax_from_PS_from_PI = (PS_from_PI + (2 * a)) / 3 if PS_from_PI else None

                values_to_insert = {
                    'PS_from_Sovera': PS_from_Sovera,
                    'PS_from_PI': PS_from_PI,
                    'Sovera_from_PS_from_PI': Sovera_from_PS_from_PI,
                    'PI_from_PS_from_Sovera': PI_from_PS_from_Sovera,
                    'TAmax_from_PS_from_Sovera': TAmax_from_PS_from_Sovera,
                    'TAmax_from_PS_from_PI': TAmax_from_PS_from_PI
                }
                for key, value in values_to_insert.items():
                    comparison_dataframe.loc[key, 'Third_calc'] = value
                    # Check closeness and store conditions met in a list

                conditions_met = metric_comparison(comparison_dataframe, 3)
                # now check the conditions again:
                if len(conditions_met) > 0:
                    row_name = conditions_met[0]

                    parts = row_name.split('_from_')
                    desired_row_name = parts[1] + '_from_' + parts[2]
                    new_value = comparison_dataframe.loc[desired_row_name, 'Third_calc']
                    # We have calculated the new PS!
                    print(f"Recalculated PS (from PI): {new_value}")
                    df.loc[df['Word'] == 'DV-S', 'Value'] = new_value
                    PS = df.loc[df['Word'] == 'DV-S', 'Value'].values[0]
                    a = df.loc[df['Word'] == 'DV-a', 'Value'].values[0]
                    # Find S/D
                    df.loc[df['Word'] == 'DV-S/a', 'Value'] = round(PS / a, 2)
                    # Find PI
                    df.loc[df['Word'] == 'DV-PI', 'Value'] = round((PS - a) / ((PS + a) / 2), 2)
                    # Find TAmax
                    df.loc[df['Word'] == 'DV-TAmax', 'Value'] = round((PS + (2 * a)) / 3, 2)

            elif len(conditions_met) > 0:

                row_name = conditions_met[0]

                parts = row_name.split('_from_')
                desired_row_name = parts[1] + '_from_' + parts[2]
                new_value = comparison_dataframe.loc[desired_row_name, 'Second_calc']
                # We have calculated the new PS!
                print(f"Recalculated PS (from PI): {new_value}")
                df.loc[df['Word'] == 'DV-a', 'Value'] = round(new_value, 2)
                PS = df.loc[df['Word'] == 'DV-S', 'Value'].values[0]
                a = df.loc[df['Word'] == 'DV-a', 'Value'].values[0]
                # Find S/D
                df.loc[df['Word'] == 'DV-S/a', 'Value'] = round(PS / a, 2)
                # Find PI
                df.loc[df['Word'] == 'DV-PI', 'Value'] = round((PS - a) / ((PS + a) / 2), 2)
                # Find TAmax
                df.loc[df['Word'] == 'DV-TAmax', 'Value'] = round((PS + (2 * a)) / 3, 2)

        except ZeroDivisionError:
            print("Error: Division by zero encountered. Check the extracted values.")
    elif len(conditions_met) < 3:
        # At least 1 of the metrics is consistent, therefore PS and a can be assumed to be correct,
        # Calculate the inconsistent metrics from the PS and a calculations
        print("At least one text extraction error, correcting...")

        if 'Sovera' not in conditions_met:
            # Find S/a
            df.loc[df['Word'] == 'DV-S/a', 'Value'] = round(PS / a, 2)
            df.loc[df['Word'] == 'DV-a/S', 'Value'] = round(a / PS, 2)
        if 'PI' not in conditions_met:
            # Find PI
            df.loc[df['Word'] == 'DV-PI', 'Value'] = round((PS - a) / ((PS + a) / 2), 2)
        if 'TAmax' not in conditions_met:
            # Find TAmax
            df.loc[df['Word'] == 'DV-TAmax', 'Value'] = round((PS + (2 * a)) / 3, 2)
    else:
        print("All metrics are consistent.")

    return df
