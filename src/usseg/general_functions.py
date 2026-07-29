""" A set of functions to segment and extract data from doppler ultrasound scans"""
# Python imports
import traceback
import math
import re
import logging
from collections import deque

from rapidfuzz.distance import Levenshtein
# Module imports
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from skimage import morphology, measure
import numpy as np
from skimage.measure import find_contours
from skimage.draw import polygon_perimeter
import cv2
from PIL import Image, ImageDraw
import scipy
from scipy.ndimage.filters import gaussian_filter
from scipy import signal
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, peak_widths, savgol_filter
import statistics
import scipy.linalg
from sklearn.cluster import DBSCAN
import pandas as pd
import pydicom
import pytesseract
from pytesseract import Output

from usseg.hemodynamic_indices import (
    pulsatility_index_from_ps_ed_and_mean_velocity,
    resistive_index_from_ps_ed,
    tamax_from_envelope_temporal_mean,
    tamax_from_ps_ed_approximation,
)
from usseg.digitized_comparison import (
    returned_model_cell_value,
    select_best_digitized_model_for_image,
)

logger = logging.getLogger(__file__)

# Matplotlib tab blue for morph tracing; PIL RGBA for annotated polylines (replacing cyan).
MORPH_CURVE_COLOR = "#1f77b4"
MORPH_LINE_RGBA = (31, 119, 180, 255)
RAY_CURVE_COLOR = "#d62728"
GROW_CURVE_COLOR = "#b026ff"
GROW_LINE_RGBA = (176, 38, 255, 255)

# OCR metric label families (order matters: more specific prefixes first).
METRIC_FAMILY_PREFIXES = (
    "Lt Ophthalmic",
    "Rt Ophthalmic",
    "Lt MCA",
    "Rt MCA",
    "Lt Ut",
    "Rt Ut",
    "Umb",
    "DV",
)

OPHTHALMIC_TARGET_WORDS = [
    "1 Lt Ophthalmic A. PS",
    "Lt Ophthalmic A. ED",
    "Lt Ophthalmic A. PI",
    "Lt Ophthalmic A. RI",
    "Lt Ophthalmic A. PS/ED",
    "Lt Ophthalmic A. ED/PS",
    "2 Lt Ophthalmic A. PS",
    "1 Rt Ophthalmic A. PS",
    "Rt Opthalmic A. ED",
    "Rt Ophthalmic A. PI",
    "Rt Ophthalmic A. RI",
    "Rt Ophthalmic A. PS/ED",
    "Rt Ophthalmic A. ED/PS",
    "2 Rt Ophthalmic A. PS",
]

OPHTHALMIC_TARGET_WORDS_EXTENDED = [
    "1 Lt Ophthalmic A. PS cm/s",
    "Lt Ophthalmic A. ED cm/s",
    "Lt Ophthalmic A. PI",
    "Lt Ophthalmic A. RI",
    "Lt Ophthalmic A. PS/ED",
    "Lt Ophthalmic A. ED/PS",
    "2 Lt Ophthalmic A. PS cm/s",
    "1 Rt Ophthalmic A. PS cm/s",
    "Rt Opthalmic A. ED cm/s",
    "Rt Ophthalmic A. PI",
    "Rt Ophthalmic A. RI",
    "Rt Ophthalmic A. PS/ED",
    "Rt Ophthalmic A. ED/PS",
    "2 Rt Ophthalmic A. PS cm/s",
]


def _line_matches_metric_family(line, family):
    if family == "Lt Ophthalmic":
        return "Lt Ophthalmic" in line or "Lt Opthalmic" in line
    if family == "Rt Ophthalmic":
        return "Rt Ophthalmic" in line or "Rt Opthalmic" in line
    return family in line


def _target_belongs_to_family(word, family):
    if family == "Lt Ophthalmic":
        return "Lt Ophthalmic" in word or "Lt Opthalmic" in word
    if family == "Rt Ophthalmic":
        return "Rt Ophthalmic" in word or "Rt Opthalmic" in word
    return word.startswith(family)


def _line_looks_ophthalmic(line):
    ll = line.lower()
    if any(x in ll for x in ("ophthalmic", "opthalmic", "thaimic", "thamic")):
        return True
    return _ophthalmic_metric_kind(line) is not None


def _ophthalmic_metric_kind(line):
    norm = re.sub(r"\s+", "", line.lower())
    if "ps/ed" in norm:
        return "PS/ED"
    if "ed/ps" in norm:
        return "ED/PS"
    if re.search(r"a\.?ps$|a\.\s*ps(?:\s|$)", line, re.I) or re.search(r"a\.\s*ps\b", line, re.I):
        return "PS"
    if re.search(r"a\.?ed$|a\.\s*ed(?:\s|$)", line, re.I) or re.search(r"a\.\s*ed\b", line, re.I):
        return "ED"
    if re.search(r"a\.\s*pi(?:\s|$)", line, re.I) or re.search(r"a\.?pi\b", line, re.I):
        return "PI"
    if re.search(r"a\.\s*p[l1](?:\s|$)", line, re.I):
        return "PI"
    if re.search(r"a\.\s*ri(?:\s|$)", line, re.I) or re.search(r"a\.?ri\b", line, re.I):
        return "RI"
    return None


def _detect_ophthalmic_side_from_ocr(ocr_data):
    texts = [str(t).strip() for t in ocr_data.get("text", []) if t and str(t).strip()]
    lt = sum(1 for t in texts if t == "Lt")
    rt = sum(1 for t in texts if t == "Rt")
    if rt > lt:
        return "Rt Ophthalmic"
    return "Lt Ophthalmic"


def _detect_metric_family_from_lines(lines, ocr_data=None):
    counts = {
        prefix: sum(1 for line in lines if _line_matches_metric_family(line, prefix))
        for prefix in METRIC_FAMILY_PREFIXES
    }
    ophthalmic_count = sum(1 for line in lines if _line_looks_ophthalmic(line))
    if ophthalmic_count > 0:
        side = _detect_ophthalmic_side_from_ocr(ocr_data or {})
        counts[side] = ophthalmic_count
    best = max(counts.values())
    if best == 0:
        return None
    return max(counts, key=counts.get)


def _match_ophthalmic_lines(lines, target_words):
    """Map OCR lines to ophthalmic target labels by metric suffix and PS order."""
    df = pd.DataFrame(columns=["Line", "Word", "Value", "Unit"])
    matched_lines = set()
    remaining_targets = list(target_words)
    ps_targets = [
        w for w in target_words
        if re.search(r"A\.\s*PS$", w) and "PS/" not in w and "ED/" not in w
    ]
    ps_idx = 0
    value_re = re.compile(r"(\-?\d+(\s*\d+)*\.\s*\d+|\-?\d+(\s*\d+)*)\s*([^\d\s]+)?$")

    def extract_value_unit(line):
        match = value_re.search(line)
        if not match:
            return None, ""
        return float(match.group(1).replace(" ", "")), (match.group(4) or "")

    def pop_target_for_kind(kind):
        nonlocal ps_idx
        if kind == "PS":
            if ps_idx < len(ps_targets):
                word = ps_targets[ps_idx]
                ps_idx += 1
                return word
            return next((w for w in remaining_targets if re.search(r"A\.\s*PS$", w) and "PS/" not in w), None)
        if kind == "PS/ED":
            return next((w for w in remaining_targets if "PS/ED" in w), None)
        if kind == "ED/PS":
            return next((w for w in remaining_targets if "ED/PS" in w), None)
        if kind == "ED":
            return next((w for w in remaining_targets if re.search(r"A\.\s*ED$", w)), None)
        if kind == "PI":
            return next((w for w in remaining_targets if re.search(r"A\.\s*PI$", w)), None)
        if kind == "RI":
            return next((w for w in remaining_targets if re.search(r"A\.\s*RI$", w)), None)
        return None

    for i, line in enumerate(lines):
        kind = _ophthalmic_metric_kind(line)
        if kind is None:
            continue
        value, unit = extract_value_unit(line)
        if value is None:
            continue
        word = pop_target_for_kind(kind)
        if word is None:
            continue
        df.loc[len(df)] = {"Line": i + 1, "Word": word, "Value": value, "Unit": unit}
        if word in remaining_targets:
            remaining_targets.remove(word)
        matched_lines.add(i)

    # Attach orphan numeric-only lines (e.g. "20.43 cm/s") to PS rows still at zero.
    for i, line in enumerate(lines):
        if i in matched_lines or _ophthalmic_metric_kind(line) is not None:
            continue
        value, unit = extract_value_unit(line)
        if value is None:
            continue
        ps_zero = df.index[(df["Word"].isin(ps_targets)) & (df["Value"] == 0)]
        if len(ps_zero) > 0:
            row_idx = ps_zero[0]
            df.loc[row_idx, "Value"] = value
            if unit:
                df.loc[row_idx, "Unit"] = unit
            df.loc[row_idx, "Line"] = i + 1
            matched_lines.add(i)

    return df, matched_lines, remaining_targets


def _is_ophthalmic_df(df):
    if df is None or df.empty or "Word" not in df.columns:
        return False
    words = df["Word"].astype(str)
    return (
        words.str.contains("Ophthalmic", na=False).any()
        or words.str.contains("Opthalmic", na=False).any()
    )


def _df_word_metric_mask(words_series, metric):
    """Row mask for a metric token; ophthalmic labels need tighter patterns than substring."""
    if words_series.str.contains("Ophthalmic|Opthalmic", regex=True, na=False).any():
        patterns = {
            "PS": r"A\.\s*PS$",
            "ED": r"A\.\s*ED$",
            "PI": r"A\.\s*PI$",
            "RI": r"A\.\s*RI$",
            "S/D": r"PS/ED$",
            "PS/ED": r"PS/ED$",
            "ED/PS": r"ED/PS$",
            "TA": r"TAmax",
            "HR": r"HR",
        }
        pat = patterns.get(metric, metric)
        return words_series.str.contains(pat, regex=True, na=False)
    return words_series.str.contains(metric, na=False)


# Foot-based beat detection tuning (used for feet + debug overlays).
# Defaults chosen to match scratch `mean_wave_test.py`.
FOOT_SEARCH_FRACTION = 0.50
FOOT_DERIV_SMOOTH_WINDOW_MAX = 11
FOOT_DERIV_SMOOTH_POLYORDER = 2
FOOT_MAX_REL_HEIGHT = 0.55
FOOT_MIN_SAMPLES_BEFORE_PEAK = 3
# Avoid derivative edge artifacts when picking d2 upper bound.
FOOT_DERIV_EDGE_GUARD = 2
# Debug toggle: disable SQI beat rejection when False.
USE_SQI_FILTER = False
# Keep saved Figure 2 clean by default (used in HTML output).
SHOW_BEAT_DEBUG_SUBPLOTS = False
# Region-grow: main-steps figure (+ optional detailed pipeline); not saved to batch.
SHOW_GROW_DEBUG_PLOTS = True
# Morphological method: refined-mask stages + Method-1 envelope overlay.
SHOW_MORPH_DEBUG_PLOTS = True
# Ray tracing: Method-2 main-steps figure (k-means → yellow → picks → smooth).
SHOW_RAY_DEBUG_PLOTS = True
# Disk radius (pixels) for binary erosion of the refined mask before region-grow.
# 0 disables. Shrinking the seed avoids over-thick refined blobs dominating seed
# statistics and lets growth fill troughs; if erosion removes all seeds, the
# full refined∩allowed seed is used instead.
GROW_SEED_EROSION_RADIUS = 10
# Region-grow intensity (normalized [0,1]): wider = accept dimmer neighbors (jittery tops).
GROW_INTENSITY_TOL = 0.43
# Hard floor: seed_mean - max(std * mult, min_drop_below_mean); autocomputed value
# is clamped to at least GROW_MIN_INTENSITY_AUTOCOMPUTE_FLOOR (not below 0.05).
GROW_MIN_FLOOR_STD_MULT = 5.0
GROW_MIN_FLOOR_MIN_DROP = 0.40
GROW_MIN_INTENSITY_AUTOCOMPUTE_FLOOR = 0.05
# Running-mean band half-width uses max(std * mult, intensity_tol) on each side.
GROW_RUNNING_STD_MULT = 3.9
# Chebyshev radius per BFS step: 1 = 8-neighbour (if connectivity=8); 2 = 5×5−1
# neighbours (24), bridging a 1-pixel gap in one hop. Larger = faster fill, more leak risk.
GROW_NEIGHBOUR_CHEBYSHEV_RADIUS = 2

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


def allowed_mask_from_roi_bounds(h: int, w: int, Xmin, Xmax, Ymin, Ymax) -> np.ndarray:
    """
    Boolean (H, W) mask of columns/rows kept after the same ROI cropping as
    ``refine_waveform_segmentation`` (X / Y clamps and Ymin-50 top margin).
    """
    allowed = np.zeros((h, w), dtype=bool)
    x0 = max(0, int(Xmin) - 1)
    x1 = min(w, int(Xmax))
    y0 = max(0, int(Ymin) - 50)
    y1 = min(h, int(Ymax))
    if x1 > x0 and y1 > y0:
        allowed[y0:y1, x0:x1] = True
    return allowed


def _shrink_seed_for_region_grow(refined_bool, allowed_bool, erosion_radius: int):
    """
    Erode the refined binary mask, then intersect with ``allowed``. If the
    result is empty, return ``refined_bool & allowed_bool`` (and log a warning).
    """
    refined_bool = np.asarray(refined_bool, dtype=bool)
    allowed_bool = np.asarray(allowed_bool, dtype=bool)
    fallback = refined_bool & allowed_bool
    if erosion_radius <= 0:
        return fallback
    foot = morphology.disk(int(erosion_radius))
    eroded = morphology.erosion(refined_bool, footprint=foot)
    shrunk = eroded & allowed_bool
    if not np.any(shrunk):
        logger.warning(
            "region grow: seed erosion (disk radius=%s) removed all seeds; "
            "using full refined∩allowed",
            erosion_radius,
        )
        return fallback
    return shrunk


def _normalize_gray_full_image_for_grow(gray: np.ndarray) -> np.ndarray:
    """Global min/max normalization to ``[0, 1]``, same as ``constrained_region_grow``."""
    img = np.asarray(gray, dtype=np.float32)
    img_min = float(np.min(img))
    img_max = float(np.max(img))
    if img_max > img_min:
        return (img - img_min) / (img_max - img_min)
    return np.zeros_like(img, dtype=np.float32)


def _region_grow_initial_intensity_params(
    gray,
    seed_mask,
    intensity_tol=None,
    min_intensity=None,
):
    """
    First-iteration intensity gates matching ``constrained_region_grow`` (before
    running-mean updates). ``seed_mask`` is boolean, full image shape.
    """
    if intensity_tol is None:
        intensity_tol = GROW_INTENSITY_TOL
    img = _normalize_gray_full_image_for_grow(gray)
    seed_mask = np.asarray(seed_mask, dtype=bool)
    if not np.any(seed_mask):
        return None
    seed_vals = img[seed_mask]
    seed_mean = float(np.mean(seed_vals))
    seed_std = float(np.std(seed_vals))
    if min_intensity is None:
        min_intensity = max(
            seed_mean
            - max(
                GROW_MIN_FLOOR_STD_MULT * seed_std,
                GROW_MIN_FLOOR_MIN_DROP,
            ),
            GROW_MIN_INTENSITY_AUTOCOMPUTE_FLOOR,
        )
    lower_intensity = max(seed_mean - intensity_tol, min_intensity)
    upper_intensity = min(seed_mean + intensity_tol, 1.0)
    return {
        "img_norm": img,
        "seed_mean": seed_mean,
        "seed_std": seed_std,
        "min_intensity": float(min_intensity),
        "lower_intensity": lower_intensity,
        "upper_intensity": upper_intensity,
        "intensity_tol": float(intensity_tol),
    }


def _bottom_most_row_per_col(mask_2d) -> np.ndarray:
    """Per column, largest row index where ``mask`` is True; NaN if column empty."""
    m = np.asarray(mask_2d, dtype=bool)
    h, w = m.shape
    rows = np.arange(h, dtype=float)[:, np.newaxis]
    masked = np.where(m, rows, np.nan)
    return np.nanmax(masked, axis=0)


def _region_grow_neighbour_offsets(connectivity: int, chebyshev_radius: int):
    """
    Integer (dy, dx) for one flood-fill layer. Chebyshev ball: all cells with
    ``0 < max(|dy|,|dx|) <= R`` (a square of side ``2R+1`` minus the centre).

    For ``R == 1``, ``connectivity`` 4 or 8 selects the usual 4- or 8-neighbour
    set. For ``R >= 2``, ``connectivity`` is ignored (full square).
    """
    r = int(chebyshev_radius)
    if r < 1:
        raise ValueError("chebyshev_radius must be >= 1")
    if r == 1:
        if connectivity == 4:
            return ((-1, 0), (1, 0), (0, -1), (0, 1))
        if connectivity == 8:
            return (
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            )
        raise ValueError("connectivity must be 4 or 8 when chebyshev_radius == 1")
    offs = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy == 0 and dx == 0:
                continue
            if max(abs(dy), abs(dx)) <= r:
                offs.append((dy, dx))
    return tuple(offs)


def constrained_region_grow(
    image: np.ndarray,
    initial_segmentation: np.ndarray,
    allowed_mask: np.ndarray,
    connectivity: int = 8,
    chebyshev_radius=None,
    intensity_tol=None,
    update_seed_stats: bool = True,
    max_area_growth: float = 3.0,
    min_intensity=None,
    forbidden_mask=None,
) -> np.ndarray:
    """
    Region growing from a seed mask, constrained to ``allowed_mask`` and intensity
    similarity to the seed. Image is normalized to [0, 1] for thresholds.

    If ``forbidden_mask`` is True at a pixel, that pixel may still be part of the
    initial seed, but the region will not **expand** into it (e.g. instrument
    yellow overlay).

    Returns a uint8 binary mask (0/1), same shape as ``image``.

    ``intensity_tol`` defaults to ``GROW_INTENSITY_TOL`` when None.
    ``chebyshev_radius`` defaults to ``GROW_NEIGHBOUR_CHEBYSHEV_RADIUS`` when None
    (1 = one pixel step; 2 = up to two in Chebyshev distance per step).
    """
    if intensity_tol is None:
        intensity_tol = float(GROW_INTENSITY_TOL)
    if chebyshev_radius is None:
        chebyshev_radius = int(GROW_NEIGHBOUR_CHEBYSHEV_RADIUS)
    else:
        chebyshev_radius = int(chebyshev_radius)
    if image.ndim != 2:
        raise ValueError("image must be a 2D grayscale array")
    if initial_segmentation.shape != image.shape:
        raise ValueError("initial_segmentation must match image shape")
    if allowed_mask.shape != image.shape:
        raise ValueError("allowed_mask must match image shape")
    if forbidden_mask is not None and forbidden_mask.shape != image.shape:
        raise ValueError("forbidden_mask must match image shape")
    if chebyshev_radius == 1 and connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8 when chebyshev_radius == 1")

    h, w = image.shape
    expand_ok = (
        allowed_mask & (~forbidden_mask)
        if forbidden_mask is not None
        else allowed_mask
    )
    img = _normalize_gray_full_image_for_grow(image)

    seed_mask = (initial_segmentation > 0) & allowed_mask
    if not np.any(seed_mask):
        return np.zeros_like(initial_segmentation, dtype=np.uint8)

    seed_vals = img[seed_mask]
    seed_mean = float(np.mean(seed_vals))
    seed_std = float(np.std(seed_vals))
    seed_area = int(np.sum(seed_mask))
    max_area = max(int(seed_area * max_area_growth), seed_area + 1)

    if min_intensity is None:
        min_intensity = max(
            seed_mean
            - max(
                GROW_MIN_FLOOR_STD_MULT * seed_std,
                GROW_MIN_FLOOR_MIN_DROP,
            ),
            GROW_MIN_INTENSITY_AUTOCOMPUTE_FLOOR,
        )

    lower_intensity = max(seed_mean - intensity_tol, min_intensity)
    upper_intensity = min(seed_mean + intensity_tol, 1.0)

    grown = seed_mask.copy()
    grown_count = seed_area
    visited = np.zeros((h, w), dtype=bool)
    q = deque()
    for y, x in np.argwhere(seed_mask):
        q.append((int(y), int(x)))
        visited[y, x] = True

    neighbours = _region_grow_neighbour_offsets(connectivity, chebyshev_radius)

    running_sum = float(np.sum(seed_vals))
    running_sq_sum = float(np.sum(seed_vals**2))
    running_n = float(seed_vals.size)

    while q:
        y, x = q.popleft()
        for dy, dx in neighbours:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            if visited[ny, nx]:
                continue
            visited[ny, nx] = True
            if not expand_ok[ny, nx] or grown[ny, nx]:
                continue

            val = float(img[ny, nx])
            if val < min_intensity:
                continue

            if update_seed_stats and running_n > 1:
                current_mean = running_sum / running_n
                current_var = max(
                    (running_sq_sum / running_n) - current_mean**2, 0.0
                )
                current_std = float(np.sqrt(current_var))
                local_lower = max(
                    current_mean
                    - max(GROW_RUNNING_STD_MULT * current_std, intensity_tol),
                    min_intensity,
                )
                local_upper = min(
                    current_mean + max(GROW_RUNNING_STD_MULT * current_std, intensity_tol),
                    1.0,
                )
            else:
                local_lower = lower_intensity
                local_upper = upper_intensity

            if local_lower <= val <= local_upper:
                grown[ny, nx] = True
                grown_count += 1
                q.append((ny, nx))
                if update_seed_stats:
                    running_sum += val
                    running_sq_sum += val * val
                    running_n += 1.0
                if grown_count >= max_area:
                    return grown.astype(np.uint8)

    return grown.astype(np.uint8)


def _plot_region_grow_debug(
    input_image_bgr,
    grown_binary_mask,
    grow_top_curve_coords,
):
    """
    Two-panel debug figure: (left) BGR image with grow top-trace polyline;
    (right) same with semi-transparent grown-region tint and trace — similar
    spirit to annotated overlays, for interactive inspection only.
    """
    rgb = cv2.cvtColor(np.asarray(input_image_bgr), cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)

    def _overlay_trace(ax, coords):
        if coords is None or len(coords) == 0:
            return False
        arr = np.asarray(coords)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return False
        rows, cols = arr[:, 0], arr[:, 1]
        order = np.argsort(cols)
        ax.plot(
            cols[order],
            rows[order],
            color=GROW_CURVE_COLOR,
            linewidth=1.8,
            label="grow trace",
        )
        return True

    axes[0].imshow(rgb)
    axes[0].set_title("Grow trace on image", fontsize=10)
    axes[0].axis("off")
    if _overlay_trace(axes[0], grow_top_curve_coords):
        axes[0].legend(loc="upper right", fontsize=8)

    axes[1].imshow(rgb)
    axes[1].set_title("Grown mask + trace", fontsize=10)
    axes[1].axis("off")
    if grown_binary_mask is not None:
        g = np.asarray(grown_binary_mask) > 0
        if np.any(g):
            tint = np.zeros((*rgb.shape[:2], 4), dtype=float)
            tint[g] = mcolors.to_rgba(GROW_CURVE_COLOR, alpha=0.28)
            axes[1].imshow(tint)
    if _overlay_trace(axes[1], grow_top_curve_coords):
        axes[1].legend(loc="upper right", fontsize=8)

    fig.suptitle("Region-grow debug", fontsize=11, y=1.02)
    fig.tight_layout()


def _plot_region_grow_pipeline_debug(
    input_image_bgr,
    gray,
    allowed_mask,
    yellow_forbidden_mask,
    refined_segmentation_mask,
    grown_binary_mask,
    Xmin,
    Xmax,
    Ymin,
    Ymax,
    grow_top_curve_coords=None,
    grow_seed_mask=None,
    intensity_tol=None,
    roi_pad: int = 24,
):
    """
    Three separate matplotlib figures (ROI zoom): (1) 3×3 mask panels, (2) troughs /
    vertical-extent overlay, (3) two intensity-vs-row profiles with gate lines.

    ``grow_seed_mask`` is the boolean mask actually passed into region growing
    (typically eroded refined∩allowed); if None, uses ``refined ∧ allowed``.
    """
    h, w = gray.shape
    x0 = max(0, int(Xmin) - 1 - roi_pad)
    x1 = min(w, int(Xmax) + roi_pad)
    y0 = max(0, int(Ymin) - 50 - roi_pad)
    y1 = min(h, int(Ymax) + roi_pad)
    x1 = max(x1, x0 + 2)
    y1 = max(y1, y0 + 2)
    sl_y = slice(y0, y1)
    sl_x = slice(x0, x1)

    def crop(a):
        return np.asarray(a)[sl_y, sl_x]

    allowed = np.asarray(allowed_mask, dtype=bool)
    refined = np.asarray(refined_segmentation_mask) > 0
    seed_full = refined & allowed
    seed_for_grow = (
        np.asarray(grow_seed_mask, dtype=bool)
        if grow_seed_mask is not None
        else seed_full
    )
    yellow = (
        np.asarray(yellow_forbidden_mask, dtype=bool)
        if yellow_forbidden_mask is not None
        else np.zeros((h, w), dtype=bool)
    )
    expand_ok = allowed & (~yellow)
    grown_full = (
        np.asarray(grown_binary_mask) > 0
        if grown_binary_mask is not None
        else np.zeros((h, w), dtype=bool)
    )
    new_growth = grown_full & (~seed_for_grow)

    if intensity_tol is None:
        intensity_tol = GROW_INTENSITY_TOL

    ip = _region_grow_initial_intensity_params(
        gray, seed_for_grow, intensity_tol=intensity_tol
    )
    if ip is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.set_title(
            "Region-grow pipeline (no seed: refined ∧ allowed is empty)", fontsize=11
        )
        ax.axis("off")
        fig.tight_layout()
        return

    img_n = ip["img_norm"]
    rgb = cv2.cvtColor(np.asarray(input_image_bgr), cv2.COLOR_BGR2RGB)[sl_y, sl_x]
    titles = [
        "RGB (ROI)",
        "Norm gray [0,1] (grower)",
        "Refined binary (morph output)",
        "Allowed (ROI box)",
        "Forbidden (yellow HSV)",
        "Expand-OK (allowed ∧ ¬yellow)",
        "Grow seed (BFS start; eroded ∩ allowed)",
        "Grown mask",
        "New pixels (grown \\ grow seed)",
    ]
    arrs = [
        rgb,
        crop(img_n),
        crop(refined.astype(float)),
        crop(allowed.astype(float)),
        crop(yellow.astype(float)),
        crop(expand_ok.astype(float)),
        crop(seed_for_grow.astype(float)),
        crop(grown_full.astype(float)),
        crop(new_growth.astype(float)),
    ]
    cmaps = [
        None,
        "viridis",
        "gray",
        "gray",
        "gray",
        "gray",
        "gray",
        "gray",
        "hot",
    ]

    nc = crop(img_n)
    w_roi = nc.shape[1]
    cols_local = np.arange(w_roi, dtype=float)
    seed_c = crop(seed_for_grow)
    grown_c = crop(grown_full)
    bs = _bottom_most_row_per_col(seed_c)
    bg = _bottom_most_row_per_col(grown_c)

    gate_txt = (
        f"Initial gates (before running-mean updates): "
        f"seed_mean={ip['seed_mean']:.3f}  seed_std={ip['seed_std']:.3f}  "
        f"intensity_tol={ip['intensity_tol']:.2f}  "
        f"min_floor={ip['min_intensity']:.3f}  "
        f"band=[{ip['lower_intensity']:.3f}, {ip['upper_intensity']:.3f}]"
    )

    # --- Figure 1: 3×3 mask grid only ---
    fig_masks, axes_m = plt.subplots(3, 3, figsize=(16, 14))
    for i in range(9):
        r, c = divmod(i, 3)
        ax = axes_m[r, c]
        if cmaps[i] is None:
            ax.imshow(arrs[i], aspect="auto")
        else:
            ax.imshow(arrs[i], cmap=cmaps[i], vmin=0, vmax=1, aspect="auto")
        ax.set_title(titles[i], fontsize=9)
        ax.axis("off")
    fig_masks.suptitle(
        f"Region-grow — masks (ROI y={y0}:{y1}, x={x0}:{x1})\n{gate_txt}",
        fontsize=10,
        y=0.995,
    )
    fig_masks.tight_layout(rect=[0, 0, 1, 0.96])

    # --- Figure 2: troughs / vertical extent (single axes) ---
    fig_trough, ax_big = plt.subplots(1, 1, figsize=(16, 8))
    ax_big.imshow(nc, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    vb = np.isfinite(bs)
    vg = np.isfinite(bg)
    if np.any(vb):
        ax_big.plot(
            cols_local[vb],
            bs[vb],
            color="red",
            linewidth=1.2,
            label="seed bottom (max row)",
        )
    if np.any(vg):
        ax_big.plot(
            cols_local[vg],
            bg[vg],
            color=GROW_CURVE_COLOR,
            linewidth=1.8,
            label="grown bottom (max row)",
        )
    if grow_top_curve_coords is not None and len(grow_top_curve_coords) > 0:
        arr = np.asarray(grow_top_curve_coords)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            rows, cols = arr[:, 0], arr[:, 1]
            m = (rows >= y0) & (rows < y1) & (cols >= x0) & (cols < x1)
            if np.any(m):
                cx = cols[m] - x0
                ry = rows[m] - y0
                order = np.argsort(cx)
                ax_big.plot(
                    cx[order],
                    ry[order],
                    color="cyan",
                    linewidth=1.0,
                    linestyle="--",
                    label="grow top trace",
                )
    ax_big.set_title(
        "Troughs / vertical extent: norm gray + per-column bottom row "
        "(seed vs grown) + grow top trace",
        fontsize=11,
    )
    ax_big.legend(loc="upper right", fontsize=8)
    ax_big.axis("off")
    fig_trough.suptitle(
        f"Region-grow — troughs (ROI y={y0}:{y1}, x={x0}:{x1})",
        fontsize=10,
        y=1.01,
    )
    fig_trough.tight_layout()

    # --- Figure 3: ROI context + intensity vs row (two columns profiled) ---
    # Pre-compute column indices: centre of ROI crop, and column where grown
    # extends furthest below seed (largest trough gap).
    xc = w_roi // 2
    gap = np.nan_to_num(bg - bs, nan=0.0)
    x_rel = int(np.argmax(gap)) if w_roi > 0 else 0
    rows_axis = np.arange(nc.shape[0], dtype=float)
    prof_c = nc[:, xc]
    prof_w = nc[:, x_rel]

    fig_prof = plt.figure(figsize=(14, 10))
    gs_prof = fig_prof.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.28, wspace=0.28)
    ax_ctx = fig_prof.add_subplot(gs_prof[0, :])
    ax_p0 = fig_prof.add_subplot(gs_prof[1, 0])
    ax_p1 = fig_prof.add_subplot(gs_prof[1, 1])

    # Context: same normalized gray as the grower; vertical lines = one-pixel-wide
    # columns that are plotted in the row below.
    ax_ctx.imshow(nc, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax_ctx.axvline(
        xc,
        color="lime",
        linewidth=2.5,
        label=f"Centre column (full x={x0 + xc})",
    )
    ax_ctx.axvline(
        x_rel,
        color="magenta",
        linewidth=2.5,
        label=f"Max (grown−seed) gap column (full x={x0 + x_rel})",
    )
    ax_ctx.set_title(
        "Where the profiles come from: ROI normalized gray [0,1] — "
        "vertical lines are the single columns sliced top→bottom below",
        fontsize=10,
    )
    ax_ctx.set_xlabel("Column within ROI crop (0 = left)")
    ax_ctx.set_ylabel("Row within ROI crop (0 = top)")
    ax_ctx.legend(loc="upper right", fontsize=8)

    # Black curve: intensity at each row along that one column. Vertical orange
    # lines on the *profile* plot = constant intensity (gates), not spatial position.
    prof_xlabel = (
        "Normalized intensity at this row\n(same [0,1] scale as region-grow)"
    )
    prof_ylabel = "Row in ROI crop (scan top → bottom of lime/magenta line)"

    ax_p0.plot(prof_c, rows_axis, color="black", linewidth=1.2, label="intensity")
    ax_p0.axvline(
        ip["lower_intensity"],
        color="orange",
        linestyle="--",
        linewidth=1,
        label="lower gate",
    )
    ax_p0.axvline(
        ip["upper_intensity"],
        color="orange",
        linestyle=":",
        linewidth=1,
        label="upper gate",
    )
    ax_p0.axvline(
        ip["min_intensity"],
        color="brown",
        linestyle="-.",
        linewidth=1,
        label="min floor",
    )
    ax_p0.axvline(ip["seed_mean"], color="lime", linewidth=1, label="seed mean")
    ax_p0.set_xlabel(prof_xlabel, fontsize=9)
    ax_p0.set_ylabel(prof_ylabel, fontsize=9)
    ax_p0.set_title(
        f"1D slice down lime line\n(full image x={x0 + xc})",
        fontsize=10,
    )
    ax_p0.legend(loc="best", fontsize=6)
    ax_p0.invert_yaxis()

    ax_p1.plot(prof_w, rows_axis, color="black", linewidth=1.2, label="intensity")
    ax_p1.axvline(ip["lower_intensity"], color="orange", linestyle="--", linewidth=1)
    ax_p1.axvline(ip["upper_intensity"], color="orange", linestyle=":", linewidth=1)
    ax_p1.axvline(ip["min_intensity"], color="brown", linestyle="-.", linewidth=1)
    ax_p1.axvline(ip["seed_mean"], color="lime", linewidth=1)
    if np.isfinite(bs[x_rel]):
        ax_p1.axhline(
            bs[x_rel],
            color="red",
            linestyle="--",
            linewidth=0.8,
            label="seed bottom row",
        )
    if np.isfinite(bg[x_rel]):
        ax_p1.axhline(
            bg[x_rel],
            color=GROW_CURVE_COLOR,
            linestyle="-",
            linewidth=1.0,
            label="grown bottom row",
        )
    ax_p1.set_xlabel(prof_xlabel, fontsize=9)
    ax_p1.set_ylabel(prof_ylabel, fontsize=9)
    ax_p1.set_title(
        f"1D slice down magenta line\n(full image x={x0 + x_rel})",
        fontsize=10,
    )
    ax_p1.legend(loc="best", fontsize=6)
    ax_p1.invert_yaxis()

    fig_prof.suptitle(
        f"Region-grow — intensity profiles (ROI y={y0}:{y1}, x={x0}:{x1})",
        fontsize=11,
        y=0.98,
    )
    caption = (
        "Bottom plots = one vertical column from the image above. For each row (top→bottom), "
        "normalized brightness [0,1] is plotted on the horizontal axis; row index on the vertical. "
        "Orange vertical lines are intensity gates (not x-positions in the image). "
        "Where the black curve falls outside the band, that pixel fails the initial grow test."
    )
    fig_prof.text(0.5, 0.012, caption, ha="center", va="bottom", fontsize=8)
    fig_prof.tight_layout(rect=[0, 0.08, 1, 0.94])


def _morph_roi_slices(h, w, Xmin, Xmax, Ymin, Ymax, roi_pad=24):
    """ROI crop slices matching refine / grow debug framing (Ymin-50 top margin)."""
    x0 = max(0, int(Xmin) - 1 - roi_pad)
    x1 = min(w, int(Xmax) + roi_pad)
    y0 = max(0, int(Ymin) - 50 - roi_pad)
    y1 = min(h, int(Ymax) + roi_pad)
    x1 = max(x1, x0 + 2)
    y1 = max(y1, y0 + 2)
    return slice(y0, y1), slice(x0, x1), y0, y1, x0, x1


def _plot_morph_refine_debug(input_image_bgr, stages, Xmin, Xmax, Ymin, Ymax, roi_pad=24):
    """
    Morph refine pipeline stages (pipeline_overview Stage 4 /
    ``refine_waveform_segmentation``). One 2×4 ROI-cropped figure.
    """
    gray = stages.get("gray")
    if gray is None:
        return
    h, w = gray.shape[:2]
    sl_y, sl_x, y0, y1, x0, x1 = _morph_roi_slices(
        h, w, Xmin, Xmax, Ymin, Ymax, roi_pad=roi_pad
    )

    def crop(a):
        arr = np.asarray(a)
        if arr.ndim == 2:
            return arr[sl_y, sl_x]
        return arr[sl_y, sl_x, ...]

    rgb = cv2.cvtColor(np.asarray(input_image_bgr), cv2.COLOR_BGR2RGB)

    titles = [
        "RGB (ROI)",
        "Greyscale",
        "Threshold @ 30",
        "ROI cropped (Ymin−50)",
        "Clean: rm small / holes / 2×erode / dilate",
        "Close + holes999 + median 3×3",
        "After CC filter (keep large)",
        "Final (Gaussian σ=7 → >0.5)",
    ]
    arrs = [
        crop(rgb),
        crop(stages["gray"]),
        crop(stages["threshold_30"]),
        crop(stages["roi_cropped"]),
        crop(stages["after_clean1"]),
        crop(stages["after_close_median"]),
        crop(stages["after_cc_filter"]),
        crop(stages["final"]),
    ]
    cmaps = [None, "gray", "gray", "gray", "gray", "gray", "gray", "gray"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for i, ax in enumerate(axes.ravel()):
        a = arrs[i]
        if cmaps[i] is None:
            ax.imshow(a, aspect="auto")
        else:
            # Binary / intensity panels: normalise display to [0,1]
            a_f = np.asarray(a, dtype=float)
            vmax = float(np.max(a_f)) if a_f.size else 1.0
            if vmax > 1.0:
                a_f = a_f / vmax
            ax.imshow(a_f, cmap=cmaps[i], vmin=0, vmax=1, aspect="auto")
        ax.set_title(titles[i], fontsize=9)
        ax.axis("off")
    fig.suptitle(
        f"Morph — refine_waveform_segmentation "
        f"(ROI y={y0}:{y1}, x={x0}:{x1}; bounds "
        f"X=[{int(Xmin)},{int(Xmax)}] Y=[{int(Ymin)},{int(Ymax)}])",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])


def _plot_morph_envelope_debug(
    input_image_bgr,
    refined_mask,
    eroded,
    outline_raw,
    outline_sided,
    top_curve_mask,
    top_curve_coords,
    keep,
    y_zero=None,
    Xmin=None,
    Xmax=None,
    Ymin=None,
    Ymax=None,
    roi_pad=24,
):
    """
    Method 1 (pipeline_overview): morph only inherits the refined mask, then
    takes outline = mask − erosion, clears the non-diagnostic side, and thins.

    One figure — what matters for diagnosis:
      1) Input refined mask on RGB (errors here are inherited directly)
      2) Shell after side-clear (keep=upper/lower)
      3) Final one-pixel morph trace
    """
    refined = np.asarray(refined_mask) > 0
    outline = np.asarray(outline_sided) > 0
    h, w = refined.shape[:2]
    if Xmin is None:
        Xmin = 0
    if Xmax is None:
        Xmax = w
    if Ymin is None:
        Ymin = 0
    if Ymax is None:
        Ymax = h
    sl_y, sl_x, y0, y1, x0, x1 = _morph_roi_slices(
        h, w, Xmin, Xmax, Ymin, Ymax, roi_pad=roi_pad
    )

    def crop(a):
        arr = np.asarray(a)
        if arr.ndim == 2:
            return arr[sl_y, sl_x]
        return arr[sl_y, sl_x, ...]

    rgb = cv2.cvtColor(np.asarray(input_image_bgr), cv2.COLOR_BGR2RGB)
    rgb_c = crop(rgb)
    refined_c = crop(refined)
    outline_c = crop(outline)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)

    # 1) What comes in — the refined mask is the entire morph method's input
    axes[0].imshow(rgb_c)
    tint = np.zeros((*rgb_c.shape[:2], 4), dtype=float)
    tint[refined_c] = mcolors.to_rgba(MORPH_CURVE_COLOR, alpha=0.35)
    axes[0].imshow(tint)
    axes[0].set_title(
        "1. Input: refined mask on RGB\n(morph inherits this body)",
        fontsize=9,
    )
    axes[0].axis("off")

    # 2) Shell after clearing the non-diagnostic side
    axes[1].imshow(rgb_c)
    shell = np.zeros((*rgb_c.shape[:2], 4), dtype=float)
    shell[outline_c] = mcolors.to_rgba("lime", alpha=0.9)
    axes[1].imshow(shell)
    axes[1].set_title(
        f"2. Outline after side-clear (keep={keep})\nmask − erosion, non-diagnostic side zeroed",
        fontsize=9,
    )
    axes[1].axis("off")

    # 3) Final thinned trace
    axes[2].imshow(rgb_c)
    if top_curve_coords is not None and len(top_curve_coords) > 0:
        arr = np.asarray(top_curve_coords)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            rows, cols = arr[:, 0], arr[:, 1]
            m = (rows >= y0) & (rows < y1) & (cols >= x0) & (cols < x1)
            if np.any(m):
                order = np.argsort(cols[m])
                axes[2].plot(
                    cols[m][order] - x0,
                    rows[m][order] - y0,
                    color=MORPH_CURVE_COLOR,
                    linewidth=2.0,
                    label="morph (1 px / col)",
                )
                axes[2].legend(loc="upper right", fontsize=8)
    axes[2].set_title(
        "3. Final morph envelope\nkeep_one_pixel_per_column",
        fontsize=9,
    )
    axes[2].axis("off")

    if y_zero is not None:
        y_local = float(y_zero) - y0
        for ax in axes:
            ax.axhline(y_local, color="yellow", linestyle="--", linewidth=0.9, alpha=0.85)

    y0_txt = f"y_zero={y_zero:.1f}" if y_zero is not None else "y_zero=None"
    fig.suptitle(
        f"Morph — Method 1 envelope ({y0_txt}; ROI y={y0}:{y1}, x={x0}:{x1})",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()


def _plot_ray_main_steps_debug(
    roi_bgr,
    roi_gray,
    signal_pre_yellow,
    yellow_mask,
    signal_mask,
    picked_y_raw,
    picked_y_smooth,
    full_rgb,
    trace_mask_full,
    x1,
    x2,
    y1,
    y2,
    keep,
    max_col_step_y,
):
    """
    Method 2 (pipeline_overview): k-means signal → yellow removal → column
    picks with continuity → gap-fill / Hampel / medfilt → full-image mask.
    """
    roi_rgb = cv2.cvtColor(np.asarray(roi_bgr), cv2.COLOR_BGR2RGB)
    rows, cols = roi_gray.shape
    xs = np.arange(cols, dtype=float)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    axes[0, 0].imshow(roi_rgb)
    axes[0, 0].set_title("1. Ray ROI (inset applied)", fontsize=9)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(roi_rgb)
    sig0 = np.ma.masked_where(~np.asarray(signal_pre_yellow), np.ones_like(roi_gray))
    axes[0, 1].imshow(sig0, cmap="Greens", alpha=0.45, vmin=0, vmax=1)
    axes[0, 1].set_title("2. k-means signal (k=3, ¬background)", fontsize=9)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(roi_rgb)
    yel = np.ma.masked_where(~np.asarray(yellow_mask), np.ones_like(roi_gray))
    sig1 = np.ma.masked_where(~np.asarray(signal_mask), np.ones_like(roi_gray))
    axes[0, 2].imshow(yel, cmap="autumn", alpha=0.55, vmin=0, vmax=1)
    axes[0, 2].imshow(sig1, cmap="Greens", alpha=0.40, vmin=0, vmax=1)
    axes[0, 2].set_title("3. Yellow removed → final signal", fontsize=9)
    axes[0, 2].axis("off")

    axes[1, 0].imshow(roi_gray, cmap="gray")
    raw = np.asarray(picked_y_raw, dtype=float)
    ok = np.isfinite(raw)
    if np.any(ok):
        axes[1, 0].plot(xs[ok], raw[ok], color=RAY_CURVE_COLOR, linewidth=1.2)
    axes[1, 0].set_title(
        f"4. Column picks (keep={keep}, step≤{int(max_col_step_y)})",
        fontsize=9,
    )
    axes[1, 0].set_xlim([0, cols - 1])
    axes[1, 0].set_ylim([rows - 1, 0])

    axes[1, 1].imshow(roi_gray, cmap="gray")
    sm = np.asarray(picked_y_smooth, dtype=float)
    ok_s = np.isfinite(sm)
    if np.any(ok_s):
        axes[1, 1].plot(xs[ok_s], sm[ok_s], color=RAY_CURVE_COLOR, linewidth=1.6)
    axes[1, 1].set_title("5. After interp + Hampel + medfilt(3)", fontsize=9)
    axes[1, 1].set_xlim([0, cols - 1])
    axes[1, 1].set_ylim([rows - 1, 0])

    axes[1, 2].imshow(full_rgb)
    if trace_mask_full is not None:
        tm = np.asarray(trace_mask_full) > 0
        if np.any(tm):
            tint = np.zeros((*full_rgb.shape[:2], 4), dtype=float)
            tint[tm] = mcolors.to_rgba(RAY_CURVE_COLOR, alpha=0.85)
            axes[1, 2].imshow(tint)
    # Draw ROI box
    axes[1, 2].plot(
        [x1, x2, x2, x1, x1],
        [y1, y1, y2, y2, y1],
        color="yellow",
        linewidth=1.0,
        linestyle="--",
    )
    axes[1, 2].set_title("6. Final ray mask on full image", fontsize=9)
    axes[1, 2].axis("off")

    fig.suptitle(
        f"Ray — Method 2 main steps (ROI y={y1}:{y2}, x={x1}:{x2})",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])


def _plot_grow_main_steps_debug(
    input_image_bgr,
    refined_segmentation_mask,
    grow_seed_mask,
    grown_binary_mask,
    grow_top_curve_coords,
    Xmin,
    Xmax,
    Ymin,
    Ymax,
    yellow_forbidden_mask=None,
    roi_pad=24,
):
    """
    Method 3 (pipeline_overview): refined body → eroded seed → constrained
    grow → envelope thinned like morph. Overlay-focused main steps.
    """
    rgb = cv2.cvtColor(np.asarray(input_image_bgr), cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    sl_y, sl_x, y0, y1, x0, x1 = _morph_roi_slices(
        h, w, Xmin, Xmax, Ymin, Ymax, roi_pad=roi_pad
    )

    def crop(a):
        arr = np.asarray(a)
        if arr.ndim == 2:
            return arr[sl_y, sl_x]
        return arr[sl_y, sl_x, ...]

    refined = np.asarray(refined_segmentation_mask) > 0
    seed = (
        np.asarray(grow_seed_mask, dtype=bool)
        if grow_seed_mask is not None
        else refined
    )
    grown = (
        np.asarray(grown_binary_mask) > 0
        if grown_binary_mask is not None
        else np.zeros((h, w), dtype=bool)
    )
    new_px = grown & (~seed)
    yellow = (
        np.asarray(yellow_forbidden_mask, dtype=bool)
        if yellow_forbidden_mask is not None
        else np.zeros((h, w), dtype=bool)
    )

    rgb_c = crop(rgb)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    axes[0, 0].imshow(rgb_c)
    tint = np.zeros((*rgb_c.shape[:2], 4), dtype=float)
    tint[crop(refined)] = mcolors.to_rgba(MORPH_CURVE_COLOR, alpha=0.35)
    axes[0, 0].imshow(tint)
    axes[0, 0].set_title("1. Input: refined mask\n(grow starts from this body)", fontsize=9)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(rgb_c)
    tint = np.zeros((*rgb_c.shape[:2], 4), dtype=float)
    tint[crop(seed)] = mcolors.to_rgba("lime", alpha=0.45)
    axes[0, 1].imshow(tint)
    axes[0, 1].set_title(
        f"2. Seed (eroded ∩ allowed)\nr={GROW_SEED_EROSION_RADIUS}",
        fontsize=9,
    )
    axes[0, 1].axis("off")

    axes[0, 2].imshow(rgb_c)
    if np.any(crop(yellow)):
        yt = np.zeros((*rgb_c.shape[:2], 4), dtype=float)
        yt[crop(yellow)] = mcolors.to_rgba("orange", alpha=0.45)
        axes[0, 2].imshow(yt)
    axes[0, 2].set_title("3. Forbidden yellow (raster)", fontsize=9)
    axes[0, 2].axis("off")

    axes[1, 0].imshow(rgb_c)
    tint = np.zeros((*rgb_c.shape[:2], 4), dtype=float)
    tint[crop(grown)] = mcolors.to_rgba(GROW_CURVE_COLOR, alpha=0.40)
    axes[1, 0].imshow(tint)
    axes[1, 0].set_title("4. Grown region (BFS)", fontsize=9)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(rgb_c)
    tint = np.zeros((*rgb_c.shape[:2], 4), dtype=float)
    tint[crop(new_px)] = mcolors.to_rgba("cyan", alpha=0.55)
    axes[1, 1].imshow(tint)
    axes[1, 1].set_title("5. New pixels (grown \\ seed)\ntroughs recovered here", fontsize=9)
    axes[1, 1].axis("off")

    axes[1, 2].imshow(rgb_c)
    if grow_top_curve_coords is not None and len(grow_top_curve_coords) > 0:
        arr = np.asarray(grow_top_curve_coords)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            rows, cols = arr[:, 0], arr[:, 1]
            m = (rows >= y0) & (rows < y1) & (cols >= x0) & (cols < x1)
            if np.any(m):
                order = np.argsort(cols[m])
                axes[1, 2].plot(
                    cols[m][order] - x0,
                    rows[m][order] - y0,
                    color=GROW_CURVE_COLOR,
                    linewidth=2.0,
                    label="grow envelope",
                )
                axes[1, 2].legend(loc="upper right", fontsize=8)
    axes[1, 2].set_title("6. Grow envelope (morph thin on grown)", fontsize=9)
    axes[1, 2].axis("off")

    fig.suptitle(
        f"Grow — Method 3 main steps (ROI y={y0}:{y1}, x={x0}:{x1})",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])


def _morphological_top_curve_from_mask(
    mask_in,
    Ymin,
    Ymax,
    y_zero=None,
    morph_debug_plots=False,
    input_image_bgr=None,
    Xmin=None,
    Xmax=None,
):
    """
    Morphological envelope + one pixel per column; same logic as the first
    half of ``compute_top_curve``. Returns (top_curve_mask int, coords (N,2), keep str).

    When ``morph_debug_plots`` or module ``SHOW_MORPH_DEBUG_PLOTS`` is True and
    ``input_image_bgr`` is provided, builds Method-1 envelope debug figures.
    """
    show_debug = bool(morph_debug_plots or SHOW_MORPH_DEBUG_PLOTS)
    mask = np.asarray(mask_in, dtype=float).copy()
    labelled = measure.label(mask)
    rp = measure.regionprops(labelled)
    if len(rp) == 0:
        z = np.zeros_like(mask, dtype=int)
        return z, np.empty((0, 2), dtype=int), "upper"

    ws = morphology.erosion(mask).astype(float)
    outline_raw = mask - ws
    top_curve_mask = outline_raw.copy()
    keep = "upper"

    if y_zero is not None:
        y0 = float(y_zero)
        c_rows = np.where(np.sum(top_curve_mask, axis=1))[0]
        above = np.sum(c_rows < y0)
        below = np.sum(c_rows > y0)
        inverted = below > above
        if not inverted:
            for r in range(int(np.floor(y0)) + 1, top_curve_mask.shape[0]):
                top_curve_mask[r, :] = 0
            keep = "upper"
        else:
            for r in range(0, int(np.ceil(y0))):
                top_curve_mask[r, :] = 0
            keep = "lower"
    else:
        for r in range(int(rp[0].centroid[0]), top_curve_mask.shape[0]):
            top_curve_mask[r, :] = 0
        if check_inverted_curve(top_curve_mask, Ymax, Ymin):
            top_curve_mask = mask - ws
            for r in range(0, int(rp[0].centroid[0])):
                top_curve_mask[r, :] = 0
            keep = "lower"

    outline_sided = top_curve_mask.copy()
    top_curve_mask = keep_one_pixel_per_column(top_curve_mask, keep=keep).astype(int)
    top_curve_coords = np.column_stack(np.nonzero(top_curve_mask))

    if show_debug and input_image_bgr is not None:
        try:
            _plot_morph_envelope_debug(
                input_image_bgr,
                mask,
                ws,
                outline_raw,
                outline_sided,
                top_curve_mask,
                top_curve_coords,
                keep,
                y_zero=y_zero,
                Xmin=Xmin,
                Xmax=Xmax,
                Ymin=Ymin,
                Ymax=Ymax,
            )
        except Exception:
            logger.exception(
                "_morphological_top_curve_from_mask: morph envelope debug plot failed"
            )

    return top_curve_mask, top_curve_coords, keep


def segment_refinement(
    input_image_obj,
    Xmin,
    Xmax,
    Ymin,
    Ymax,
    y_zero=None,
    ray_max_col_step_y=None,
    grow_debug_plots=False,
    morph_debug_plots=False,
    grow_seed_erosion_radius=None,
    grow_forbid_yellow=True,
):
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
        ray_max_col_step_y (int, optional): Max vertical step in pixels between
            neighbouring columns in ray tracing. Default derives from ray ROI height.
        grow_debug_plots (bool, optional): If True, show Method-3 main-steps
            figure (refined → seed → yellow → grown → new pixels → envelope).
            Also respects module-level ``SHOW_GROW_DEBUG_PLOTS``.
        morph_debug_plots (bool, optional): If True, show matplotlib debug figures
            for the morphological path: refined-mask stages and Method-1
            envelope (input mask → outline → thinned trace). Also respects
            module-level ``SHOW_MORPH_DEBUG_PLOTS``.
        grow_seed_erosion_radius (int, optional): Disk radius in pixels for binary
            erosion of the refined mask before region-grow (0 = off). Default uses
            ``GROW_SEED_EROSION_RADIUS``.
        grow_forbid_yellow (bool, optional): If True (default), block region-grow
            expansion into HSV-detected instrument yellow (screenshots). Set False
            for DICOM, where yellow is rare and the mask can remove real signal.

    Returns:
        (tuple) : tuple containing:
            - **refined_segmentation_mask** (ndarray): A binary array mask showing the refined segmentation of the waveform (value 1) against the background (value 0).
            - **top_curve_mask** (ndarray): Morphological top-curve mask (one pixel per column after thinning).
            - **top_curve_coords** (ndarray): ``(row, column)`` coordinates for that morphological curve.
            - **ray_top_curve_mask** (ndarray or None): Ray-traced top-curve mask, same shape as the image, or None if not computed.
            - **ray_top_curve_coords** (ndarray or None): ``(row, column)`` for the ray curve, or None if not computed.
            - **grow_top_curve_mask** (ndarray or None): Top curve from region-growing refinement, or None if grow failed / empty.
            - **grow_top_curve_coords** (ndarray or None): ``(row, column)`` for the grow curve, or None.
    """

    # 1) Produce the refined binary segmentation mask
    refined_segmentation_mask = refine_waveform_segmentation(
        input_image_obj,
        Xmin,
        Xmax,
        Ymin,
        Ymax,
        morph_debug_plots=morph_debug_plots,
    )

    # 1b) Constrained region grow from refined mask (seed) within ROI bounds
    h_img, w_img = input_image_obj.shape[:2]
    gray = cv2.cvtColor(input_image_obj, cv2.COLOR_BGR2GRAY)
    allowed = allowed_mask_from_roi_bounds(h_img, w_img, Xmin, Xmax, Ymin, Ymax)
    yellow_forbidden = None
    if grow_forbid_yellow:
        try:
            yellow_forbidden = hsv_yellow_tick_mask_bgr(input_image_obj)
        except Exception:
            logger.exception("segment_refinement: HSV yellow mask for grow failed")
    erode_r = (
        GROW_SEED_EROSION_RADIUS
        if grow_seed_erosion_radius is None
        else int(grow_seed_erosion_radius)
    )
    grow_seed_bool = _shrink_seed_for_region_grow(
        np.asarray(refined_segmentation_mask) > 0,
        allowed,
        erode_r,
    )
    grow_seed_for_grow = grow_seed_bool.astype(np.float32)
    grown_u8 = None
    grown_binary_mask = None
    try:
        grown_u8 = constrained_region_grow(
            gray,
            grow_seed_for_grow,
            allowed,
            forbidden_mask=yellow_forbidden,
        )
        grown_binary_mask = grown_u8.astype(float)
    except Exception:
        logger.exception("segment_refinement: constrained_region_grow failed")

    # 2) From that mask, derive morphological, optional ray, and grow top-curve representations
    (
        top_curve_mask,
        top_curve_coords,
        ray_top_curve_mask,
        ray_top_curve_coords,
        grow_top_curve_mask,
        grow_top_curve_coords,
    ) = compute_top_curve(
        refined_segmentation_mask,
        Ymin,
        Ymax,
        y_zero=y_zero,
        input_image_obj=input_image_obj,
        Xmin=Xmin,
        Xmax=Xmax,
        plot_curve_comparison=True,
        ray_max_col_step_y=ray_max_col_step_y,
        morph_debug_plots=morph_debug_plots,
        grown_binary_mask=(
            grown_binary_mask
            if (
                grown_binary_mask is not None
                and np.any(np.asarray(grown_binary_mask) > 0)
            )
            else None
        ),
    )

    if grow_debug_plots or SHOW_GROW_DEBUG_PLOTS:
        try:
            _plot_grow_main_steps_debug(
                input_image_obj,
                refined_segmentation_mask,
                grow_seed_bool,
                grown_binary_mask,
                grow_top_curve_coords,
                Xmin,
                Xmax,
                Ymin,
                Ymax,
                yellow_forbidden_mask=yellow_forbidden,
            )
        except Exception:
            logger.exception("segment_refinement: grow main-steps debug plot failed")

    return (
        refined_segmentation_mask,
        top_curve_mask,
        top_curve_coords,
        ray_top_curve_mask,
        ray_top_curve_coords,
        grow_top_curve_mask,
        grow_top_curve_coords,
    )


def refine_waveform_segmentation(
    input_image_obj,
    Xmin,
    Xmax,
    Ymin,
    Ymax,
    morph_debug_plots=False,
):
    """
    Core segmentation routine used by ``segment_refinement``.

    This function performs the image thresholding and all morphological
    operations required to obtain the final refined binary mask of the
    waveform region. It does **not** compute the top-curve.

    When ``morph_debug_plots`` or module ``SHOW_MORPH_DEBUG_PLOTS`` is True,
    builds a multi-panel figure of the refine stages documented in
    pipeline_overview Stage 4.
    """

    show_debug = bool(morph_debug_plots or SHOW_MORPH_DEBUG_PLOTS)
    stages = {}

    # Refine segmentation to increase smoothing
    # Save output to .txt file to load later.

    image = input_image_obj
    input_image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if show_debug:
        stages["gray"] = input_image_gray.copy()

    ret, thresholded_image = cv2.threshold(input_image_gray, 30, 255, 0)
    if show_debug:
        stages["threshold_30"] = thresholded_image.copy()

    thresholded_image[:, int(Xmax): -1] = 0
    thresholded_image[:, 0: int(Xmin) - 1] = 0
    thresholded_image[0: int(Ymin) - 50, :] = 0
    thresholded_image[int(Ymax): -1, :] = 0
    main_ROI = thresholded_image  # Main ROI
    if show_debug:
        stages["roi_cropped"] = main_ROI.copy()

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
    if show_debug:
        stages["after_clean1"] = refined_segmentation_mask.astype(float).copy()

    refined_segmentation_mask = morphology.dilation(refined_segmentation_mask)
    refined_segmentation_mask = morphology.remove_small_holes(refined_segmentation_mask, max_size=999)
    refined_segmentation_mask = morphology.closing(refined_segmentation_mask)
    refined_segmentation_mask = refined_segmentation_mask.astype(int)
    refined_segmentation_mask = scipy.signal.medfilt(refined_segmentation_mask, 3)
    if show_debug:
        stages["after_close_median"] = np.asarray(refined_segmentation_mask, dtype=float).copy()

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
    if show_debug:
        stages["after_cc_filter"] = refined_segmentation_mask.copy()
    # refined_segmentation_mask[rr, cc] = 1 #set color white

    blurred = gaussian_filter(refined_segmentation_mask, sigma=7)
    refined_segmentation_mask = (blurred > 0.5) * 1
    if show_debug:
        stages["final"] = np.asarray(refined_segmentation_mask, dtype=float).copy()
        try:
            _plot_morph_refine_debug(
                image, stages, Xmin, Xmax, Ymin, Ymax
            )
        except Exception:
            logger.exception(
                "refine_waveform_segmentation: morph refine debug plot failed"
            )

    return refined_segmentation_mask


def ray_trace_roi_from_refined_mask(
    refined_segmentation_mask,
    h_img,
    w_img,
    Xmin,
    Xmax,
    Ymin,
    Ymax,
):
    """ROI (pixel bounds) for ray tracing from refined-mask extent and rough ROI."""
    nz_rows, nz_cols = np.where(refined_segmentation_mask > 0)
    if nz_rows.size > 0 and nz_cols.size > 0:
        rt_xmin = int(np.min(nz_cols))
        rt_xmax = int(np.max(nz_cols)) + 1
        rt_ymin = int(np.min(nz_rows))
        rt_ymax = int(np.max(nz_rows)) + 1

        w_box = max(1, rt_xmax - rt_xmin)
        h_box = max(1, rt_ymax - rt_ymin)
        pad_x = max(8, int(0.03 * w_box))
        pad_top = max(18, int(0.35 * h_box))
        pad_bottom = max(6, int(0.06 * h_box))

        rt_xmin = max(0, rt_xmin - pad_x)
        rt_xmax = min(w_img, rt_xmax + pad_x)
        rt_ymin = max(0, rt_ymin - pad_top)
        rt_ymax = min(h_img, rt_ymax + pad_bottom)

        inset_x = 15
        inset_y = max(2, int(0.01 * h_box))
        rt_xmin = min(rt_xmax - 1, rt_xmin + inset_x)
        rt_xmax = max(rt_xmin + 1, rt_xmax - inset_x)
        rt_ymin = min(rt_ymax - 1, rt_ymin + inset_y)
        rt_ymax = max(rt_ymin + 1, rt_ymax - inset_y)
    else:
        rt_xmin, rt_xmax = int(Xmin), int(Xmax)
        rt_ymin, rt_ymax = int(Ymin), int(Ymax)

    return rt_xmin, rt_xmax, rt_ymin, rt_ymax


def _hampel_1d(y, half_window=2, n_sigmas=3.0):
    """Replace sparse outliers; preserves coherent edges better than wide medfilt."""
    y = np.asarray(y, dtype=float)
    n = y.size
    if n == 0:
        return y
    out = y.copy()
    c = 1.4826
    for i in range(n):
        lo = max(0, i - half_window)
        hi = min(n, i + half_window + 1)
        win = y[lo:hi]
        med = float(np.median(win))
        mad = float(np.median(np.abs(win - med)))
        if mad < 1e-9:
            continue
        if abs(y[i] - med) > n_sigmas * c * mad:
            out[i] = med
    return out


def _smooth_1d_rolling_median(y, frac=0.08, min_window=5, max_window=11):
    """
    Light robust smoothing for digitized 1D signals.

    Uses a rolling median (robust to spikes) with an adaptively sized odd window.
    Intended to reduce residual column-to-column ray jitter after digitization.
    """
    if y is None:
        return y
    arr = np.asarray(y, dtype=float)
    n = int(arr.size)
    if n < min_window:
        return arr.tolist()

    window = int(round(n * frac))
    window = max(min_window, min(max_window, window))
    # Ensure odd window for symmetric centering.
    if window % 2 == 0:
        window = window - 1 if window > min_window else window + 1
        window = max(3, window)

    if window < 3:
        return arr.tolist()

    smoothed = pd.Series(arr).rolling(
        window=window, center=True, min_periods=1
    ).median().to_numpy()
    return smoothed.tolist()


def _smooth_1d_digitized_shape_preserving(
    y,
    frac=0.10,
    min_window=7,
    max_window=21,
    polyorder=3,
):
    """
    Smoother for digitized waveforms that aims to reduce jitter while preserving shape.

    Strategy:
      1) Hampel filter removes isolated spikes/outliers.
      2) Savitzky–Golay smooths while preserving local curvature/extrema better than
         moving-average / median-of-wide-window filters.
    """
    if y is None:
        return y
    arr = np.asarray(y, dtype=float)
    n = int(arr.size)
    if n < 3:
        return arr.tolist()

    # Remove sparse spikes first.
    try:
        arr_h = _hampel_1d(arr, half_window=2, n_sigmas=3.0)
    except Exception:
        arr_h = arr

    # Choose an odd window length based on series length.
    window = int(round(n * frac))
    window = max(min_window, min(max_window, window))
    if window % 2 == 0:
        window += 1
    window = min(window, n if n % 2 == 1 else n - 1)

    # Ensure window is valid for savgol.
    if window < 5:
        return _smooth_1d_rolling_median(arr_h.tolist(), frac=0.08, min_window=5, max_window=11)

    po = int(polyorder)
    po = max(2, min(po, window - 2))

    try:
        y_sg = scipy.signal.savgol_filter(arr_h, window_length=window, polyorder=po, mode="interp")
        return np.asarray(y_sg, dtype=float).tolist()
    except Exception:
        # Conservative fallback (robust, but may blunt peaks slightly)
        return _smooth_1d_rolling_median(arr_h.tolist(), frac=0.08, min_window=5, max_window=11)


def _pick_y_from_column_signal(
    band_mask,
    col_gray,
    keep,
    run3_hit_mask,
    x,
    y_low,
):
    """
    Single-column pick: 3-consecutive run (upper/lower) or brightest fallback.
    ``band_mask`` may already be restricted to a vertical window around prev_y.
    Returns ROI-local row ``y_pick`` or None.
    """
    run3 = np.convolve(
        band_mask.astype(np.uint8), np.ones(3, dtype=np.uint8), mode="valid"
    )
    run_start_idx = np.where(run3 == 3)[0]
    if run_start_idx.size > 0:
        run_centers = run_start_idx + 1
        run3_hit_mask[y_low + run_centers, x] = 1
        ri = 0 if keep == "upper" else -1
        y_local = int(run_centers[ri])
        return y_low + y_local
    band = col_gray.astype(float)
    band[~band_mask] = -1
    if np.any(band >= 0):
        mx = float(np.max(band))
        candidates = np.where(band == mx)[0]
        y_local = int(candidates[0] if keep == "upper" else candidates[-1])
        return y_low + y_local
    return None


def hsv_yellow_tick_mask_bgr(bgr: np.ndarray) -> np.ndarray:
    """
    Instrument yellow (ticks / trace overlay) in BGR images, using the same HSV
    gates and opening as ``ray_trace_waveform_segmentation``. True means yellow;
    callers typically exclude these pixels from signal or from region-growing.
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("bgr must be an HxWx3 BGR array")
    roi_hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = roi_hsv[:, :, 1].astype(np.float32)
    val = roi_hsv[:, :, 2].astype(np.float32)
    hue = roi_hsv[:, :, 0].astype(np.float32)
    hue_yellow_gate = (hue >= 18) & (hue <= 40)
    sat_gate = sat > 90
    val_gate = val > 90
    yellow_mask = hue_yellow_gate & sat_gate & val_gate
    yellow_mask = morphology.opening(yellow_mask, morphology.disk(1))
    return yellow_mask.astype(bool)


def ray_trace_waveform_segmentation(
    input_image_obj,
    Xmin,
    Xmax,
    Ymin,
    Ymax,
    max_col_step_y,
    keep="upper",
    debug_plots=False,
):
    """Waveform boundary via column-wise ray tracing on colour-clustered signal.

    ``keep`` must match ``compute_top_curve`` envelope choice: ``upper`` scans
    top-down (first signal run / uppermost fallback); ``lower`` scans for the
    bottom-most run / lowermost bright fallback (inverted waveforms).

    ``max_col_step_y`` (pixel rows, >= 3 after clamping) limits how far the trace
    may move vertically between neighbouring columns; the caller should set it
    (e.g. from ray ROI height). See ``compute_top_curve``.
    """
    image = input_image_obj
    h, w = image.shape[:2]
    x1 = max(0, min(w - 1, int(Xmin)))
    x2 = max(0, min(w, int(Xmax)))
    y1 = max(0, min(h - 1, int(Ymin)))
    y2 = max(0, min(h, int(Ymax)))

    # Final hard inset for ray-trace ROI used by BOTH processing and debug plots.
    # This guarantees edge-noise exclusion regardless of caller-side ROI math.
    inset_lr = 20
    x1 = min(x2 - 1, x1 + inset_lr)
    x2 = max(x1 + 1, x2 - inset_lr)

    if x2 <= x1 or y2 <= y1:
        return np.zeros((h, w), dtype=float)

    roi_bgr = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    roi_gray = gray[y1:y2, x1:x2]
    if roi_gray.size == 0:
        return np.zeros((h, w), dtype=float)

    # Dynamic colour prefilter:
    # 1) Cluster all ROI pixels in colour space.
    # 2) Estimate which cluster is background by luminance (darkest mean gray).
    # 3) Keep everything else as "signal candidates".
    px = roi_bgr.reshape((-1, 3)).astype(np.float32)
    k = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.2)
    _compactness, labels, _centers = cv2.kmeans(
        px, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.reshape(roi_gray.shape)
    cluster_means = []
    cluster_counts = []
    for ci in range(k):
        m = labels == ci
        cluster_means.append(float(np.mean(roi_gray[m])) if np.any(m) else np.inf)
        cluster_counts.append(int(np.sum(m)))
    bg_cluster = int(np.argmin(cluster_means))
    signal_pre_yellow = labels != bg_cluster
    signal_mask = signal_pre_yellow.copy()
    logger.info(
        "ray_trace: clusters mean_gray=%s counts=%s -> background_cluster=%s",
        [round(v, 2) if np.isfinite(v) else None for v in cluster_means],
        cluster_counts,
        bg_cluster,
    )

    # Remove yellow overlays (peak ticks / trace), same detector as region grow.
    yellow_mask = hsv_yellow_tick_mask_bgr(roi_bgr)
    signal_mask = signal_mask & (~yellow_mask)

    logger.info(
        "ray_trace: yellow excluded=%s pixels; signal remaining=%s",
        int(np.sum(yellow_mask)),
        int(np.sum(signal_mask)),
    )

    show_debug = bool(debug_plots or SHOW_RAY_DEBUG_PLOTS)

    rows, cols = roi_gray.shape
    max_col_step_y = int(max(3, int(max_col_step_y)))

    trace_mask_roi = np.zeros((rows, cols), dtype=np.uint8)
    picked_y_per_col = np.full(cols, np.nan, dtype=float)
    run3_hit_mask = np.zeros((rows, cols), dtype=np.uint8)

    # Logging counters (to avoid per-column spam)
    n_banded_attempts = 0
    n_fallback_full_col = 0
    n_fallback_clipped = 0

    prev_y = None
    for x in range(cols):
        y_low = 0
        full_band = signal_mask[:, x]
        col_gray = roi_gray[:, x]

        y_pick = None
        if prev_y is not None:
            n_banded_attempts += 1
            lo = max(0, int(round(prev_y)) - max_col_step_y)
            hi = min(rows, int(round(prev_y)) + max_col_step_y + 1)
            band_window = full_band.copy()
            band_window[:lo] = False
            band_window[hi:] = False
            y_pick = _pick_y_from_column_signal(
                band_window,
                col_gray,
                keep,
                run3_hit_mask,
                x,
                y_low,
            )

        if y_pick is None:
            y_pick = _pick_y_from_column_signal(
                full_band,
                col_gray,
                keep,
                run3_hit_mask,
                x,
                y_low,
            )
            if y_pick is not None and prev_y is not None:
                # Banded pick failed; full-column fallback succeeded.
                n_fallback_full_col += 1
                y_pick = int(
                    np.clip(y_pick, prev_y - max_col_step_y, prev_y + max_col_step_y)
                )
                # If clipping changed the pick, note that too.
                if abs(float(y_pick) - float(prev_y)) > float(max_col_step_y) + 1e-6:
                    # Defensive: should never happen due to clip, but keep counter meaningful.
                    n_fallback_clipped += 1
                # More relevant: detect whether clip actually moved the fallback.
                # (Compare unclipped vs clipped without storing extra state.)
                # We approximate by checking whether the fallback was near bounds.
                if y_pick == int(round(prev_y - max_col_step_y)) or y_pick == int(
                    round(prev_y + max_col_step_y)
                ):
                    n_fallback_clipped += 1

        if y_pick is None:
            continue

        trace_mask_roi[y_pick, x] = 1
        picked_y_per_col[x] = float(y_pick)
        prev_y = float(y_pick)

    # One concise log line when we had to fall back.
    if n_fallback_full_col > 0 or n_fallback_clipped > 0:
        logger.info(
            "ray_trace: fallback used (banded attempts=%d, full-column fallbacks=%d, clipped_fallbacks=%d) with max_col_step_y=%d",
            n_banded_attempts,
            n_fallback_full_col,
            n_fallback_clipped,
            int(max_col_step_y),
        )

    picked_y_raw = picked_y_per_col.copy()
    picked_y_smooth = np.full(cols, np.nan, dtype=float)

    # Interpolate and connect the traced points into a continuous signal.
    valid_cols = np.where(np.isfinite(picked_y_per_col))[0]
    if valid_cols.size >= 2:
        all_cols = np.arange(cols, dtype=float)
        y_interp = np.interp(all_cols, valid_cols.astype(float), picked_y_per_col[valid_cols])
        # Hampel suppresses isolated outliers; narrow medfilt limits blunting of
        # sustained slopes (e.g. sharp approach to diastolic foot) vs kernel 5.
        y_interp = _hampel_1d(y_interp, half_window=2, n_sigmas=3.0)
        y_interp = scipy.signal.medfilt(y_interp, kernel_size=3)
        picked_y_smooth = y_interp.astype(float)

        trace_mask_roi = np.zeros((rows, cols), dtype=np.uint8)
        y_idx = np.clip(np.round(y_interp).astype(int), 0, rows - 1)
        trace_mask_roi[y_idx, np.arange(cols)] = 1
    else:
        # Keep sparse picks when interpolation is not possible.
        trace_mask_roi = (trace_mask_roi > 0).astype(np.uint8)
        picked_y_smooth = picked_y_raw.copy()

    # Thicken traced line slightly and map back to full-image mask.
    trace_mask_roi = morphology.dilation(trace_mask_roi.astype(bool), morphology.disk(1))
    trace_mask = np.zeros((h, w), dtype=bool)
    trace_mask[y1:y2, x1:x2] = trace_mask_roi

    if show_debug:
        try:
            full_rgb = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
            _plot_ray_main_steps_debug(
                roi_bgr,
                roi_gray,
                signal_pre_yellow,
                yellow_mask,
                signal_mask,
                picked_y_raw,
                picked_y_smooth,
                full_rgb,
                trace_mask,
                x1,
                x2,
                y1,
                y2,
                keep,
                max_col_step_y,
            )
        except Exception:
            logger.exception("ray_trace: Method-2 main-steps debug plotting failed")

    return trace_mask.astype(float)


def compute_top_curve(
    refined_segmentation_mask,
    Ymin,
    Ymax,
    y_zero=None,
    input_image_obj=None,
    Xmin=None,
    Xmax=None,
    plot_curve_comparison=True,
    ray_max_col_step_y=None,
    grown_binary_mask=None,
    morph_debug_plots=False,
):
    """
    Given a refined segmentation mask, compute top-curve representations.

    Always produces the **morphological** envelope (``top_curve_mask`` /
    ``top_curve_coords``). When ``input_image_obj`` and rough ``Xmin``/``Xmax``
    are provided, also runs **ray tracing** and returns ``ray_top_curve_mask`` /
    ``ray_top_curve_coords`` (or ``None`` if ray fails or yields an empty mask).
    When ``grown_binary_mask`` is provided and non-empty, also produces
    ``grow_top_curve_mask`` / ``grow_top_curve_coords`` via the same envelope
    thinning as morph.
    Envelope choice ``keep`` (upper vs lower) is shared for thinning and for
    the ray tracer scan direction.

    Ray tracing receives ``max_col_step_y`` derived from the ray ROI height
    (``max(12, ray_roi_rows // 25)``), unless ``ray_max_col_step_y`` is set.

    If ``plot_curve_comparison`` is True, builds two figures: both coordinate
    traces on one image, and a two-panel view of the morph vs ray masks.

    If ``morph_debug_plots`` or ``SHOW_MORPH_DEBUG_PLOTS`` is True, Method-1
    envelope debug figures are built when ``input_image_obj`` is available
    (refine-stage plots are produced earlier by ``refine_waveform_segmentation``).
    """
    top_curve_mask, top_curve_coords, keep = _morphological_top_curve_from_mask(
        refined_segmentation_mask,
        Ymin,
        Ymax,
        y_zero=y_zero,
        morph_debug_plots=morph_debug_plots,
        input_image_bgr=input_image_obj,
        Xmin=Xmin,
        Xmax=Xmax,
    )

    grow_top_curve_mask = None
    grow_top_curve_coords = None
    if grown_binary_mask is not None and np.any(np.asarray(grown_binary_mask) > 0):
        grow_top_curve_mask, grow_top_curve_coords, _ = (
            _morphological_top_curve_from_mask(
                grown_binary_mask,
                Ymin,
                Ymax,
                y_zero=y_zero,
                morph_debug_plots=False,
            )
        )

    ray_top_curve_mask = None
    ray_top_curve_coords = None
    if input_image_obj is not None and Xmin is not None and Xmax is not None:
        h_img, w_img = input_image_obj.shape[:2]
        rtx0, rtx1, rty0, rty1 = ray_trace_roi_from_refined_mask(
            refined_segmentation_mask, h_img, w_img, Xmin, Xmax, Ymin, Ymax
        )
        ray_roi_rows = max(1, int(rty1) - int(rty0))
        max_col_step_y = (
            int(ray_max_col_step_y)
            if ray_max_col_step_y is not None
            else max(30, ray_roi_rows // 25)
        )
        try:
            ray_mask = ray_trace_waveform_segmentation(
                input_image_obj,
                rtx0,
                rtx1,
                rty0,
                rty1,
                max_col_step_y,
                keep=keep,
                debug_plots=False,
            )
            if np.any(ray_mask > 0):
                ray_thin = keep_one_pixel_per_column(
                    ray_mask.astype(bool), keep=keep
                ).astype(int)
                ray_top_curve_mask = ray_thin
                ray_top_curve_coords = np.column_stack(np.nonzero(ray_thin))
        except Exception:
            logger.exception("compute_top_curve: ray trace failed")

    if plot_curve_comparison:
        try:
            def _sorted_xy(coords):
                if coords is None or len(coords) == 0:
                    return None, None
                arr = np.asarray(coords)
                if arr.ndim != 2 or arr.shape[1] < 2:
                    return None, None
                rows, cols = arr[:, 0], arr[:, 1]
                order = np.argsort(cols)
                return cols[order], rows[order]

            if input_image_obj is not None:
                rgb = cv2.cvtColor(
                    np.asarray(input_image_obj), cv2.COLOR_BGR2RGB
                )
                fig_c, ax_c = plt.subplots(1, 1, figsize=(10, 6))
                ax_c.imshow(rgb)
                xm, ym = _sorted_xy(top_curve_coords)
                if xm is not None:
                    ax_c.plot(xm, ym, color="lime", linewidth=1.2, label="morph")
                xr, yr = _sorted_xy(ray_top_curve_coords)
                if xr is not None:
                    ax_c.plot(xr, yr, color="red", linewidth=1.2, label="ray")
                xg, yg = _sorted_xy(grow_top_curve_coords)
                if xg is not None:
                    ax_c.plot(xg, yg, color="magenta", linewidth=1.0, label="grow")
                ax_c.legend(loc="upper right")
                ax_c.set_title("top_curve_coords vs ray_top_curve_coords")
                ax_c.axis("off")
                fig_c.tight_layout()

            fig_m, axes_m = plt.subplots(
                1, 2, figsize=(12, 5), sharex=True, sharey=True
            )
            axes_m[0].imshow(np.asarray(top_curve_mask) > 0, cmap="gray", vmin=0, vmax=1)
            axes_m[0].set_title("top_curve_mask (morph)")
            axes_m[0].axis("off")
            if ray_top_curve_mask is not None:
                axes_m[1].imshow(
                    np.asarray(ray_top_curve_mask) > 0,
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                )
                axes_m[1].set_title("ray_top_curve_mask")
            else:
                axes_m[1].imshow(
                    np.zeros_like(np.asarray(top_curve_mask), dtype=float),
                    cmap="gray",
                )
                axes_m[1].set_title("ray_top_curve_mask (none)")
            axes_m[1].axis("off")
            fig_m.suptitle("Curve masks", y=1.02)
            fig_m.tight_layout()
        except Exception:
            logger.exception("compute_top_curve: comparison plotting failed")

    return (
        top_curve_mask,
        top_curve_coords,
        ray_top_curve_mask,
        ray_top_curve_coords,
        grow_top_curve_mask,
        grow_top_curve_coords,
    )


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
        W_right = morphology.remove_small_objects(W_right, max_size=5, connectivity=2) # SAMSUNG - REMOVES TICKS, MAX_SIZE @15 TOO HIGH?
        
        # Label connected components
        labels = measure.label(W_right, connectivity=2)
        props = measure.regionprops(labels)
        
        # Define size thresholds for tick-like objects
        min_tick_area = 5  # Minimum area for a tick
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
                    ]  # Left axis ROI
        elif Side == "Right":
            ROIAX = thresholded_image[
                    int(Right_dimensions[2]): int(Right_dimensions[3]),
                    int(Right_dimensions[0] + TYLshift): int(Right_dimensions[1]),
                    ]  # Right axis ROI

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

    # Build label candidates: (text, box centre) pairs, dropping "" and "-" together
    n_boxes = len(extracted_text_data["level"])
    CenBox = []      # centres of boxes to keep
    label_texts = [] # texts for boxes to keep
    for i in range(1, n_boxes):  # dont start from 0 as the first index is redundant
        txt_val = extracted_text_data["text"][i]

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

        # De-duplicate Tesseract repeated rows
        o = i / 4  # we get 4 repeats for each real box, so this reduces that to 1.
        if not o.is_integer():
            continue

        # Drop empty/standalone minus labels together with their boxes so that
        # numbers and positions stay aligned.
        if txt_val == "" or txt_val == "-":
            continue

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

        label_texts.append(txt_val)

        cv2.rectangle(
            ROI3, (x, y), (x + wi, y + h), (255), 2
        )  # Draw the rectangles on the ROI for retained labels

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

    # Final numbers are derived directly from label_texts so they stay aligned
    # with CenBox/positions.
    number = [correct_number_format(s) for s in label_texts]

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
            logger.info(
                "Axis validation for %s side: could not convert tick value '%s' to float; skipping this tick.",
                side,
                v,
            )

    numbers = valid_numbers
    positions = valid_positions

    if len(numbers) < 2 or len(numbers) != len(positions):
        logger.info(
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
                    logger.info(
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
                    logger.info(
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
                logger.info(
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
            logger.info(
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
                    logger.info(
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

        # If bad slopes remain, we can't safely decide which tick(s) to drop.
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


def plot_digitized_data_single_axis(
    Rticks,
    Rlocs,
    Lticks,
    Llocs,
    top_curve_coords,
    overlay_curve_coords=None,
    overlay_is_ray=False,
    grow_curve_coords=None,
):
    """
    Simplified digitization using a single vertical axis and a normalized X axis.

    - Chooses one "best" vertical axis (Left or Right) for value calibration.
      If both sides are present and broadly agree, uses the side with more ticks.
      If only one side is usable, falls back to that side.
    - Uses that axis to build a linear mapping from pixel y to physical value.
    - Uses curve x-position (or index) to build an arbitrary normalized time axis [0, 1].

    Args:
        Rticks, Rlocs, Lticks, Llocs, top_curve_coords: as for ``plot_digitized_data``.
        overlay_curve_coords: optional second ``(row,col)`` curve for comparison.
        overlay_is_ray: if True, overlay is ray (red) and main is morph (blue);
            if False, overlay is morph (blue) and main is ray (red).
        grow_curve_coords: optional third ``(row,col)`` region-grow curve (violet).

    Returns:
        Xplot, Yplot, Ynought: same semantics as ``plot_digitized_data`` (from ``top_curve_coords`` only).
        Xplot_overlay, Yplot_overlay: second series for comparison plots (empty lists if no overlay).
        Xplot_grow, Yplot_grow: region-grow series (empty lists if no grow curve).
    """

    # Convert to lists defensively
    Rticks = list(Rticks) if Rticks is not None else []
    Rlocs = [list(p) for p in Rlocs] if Rlocs is not None else []
    Lticks = list(Lticks) if Lticks is not None else []
    Llocs = [list(p) for p in Llocs] if Llocs is not None else []

    # If both sides are completely empty, we cannot digitize
    if not Rticks and not Lticks:
        logger.error("Digitization: both axes empty - cannot digitize waveform.")
        return [], [], [0], [], [], [], []

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
        return [], [], [0], [], [], [], []

    # Build a simple linear mapping from pixel y to value using the chosen axis
    # Axis locations are [x, y]; we care about y here.
    try:
        pairs = [(float(p[1]), float(v)) for v, p in zip(axis_ticks, axis_locs)]
    except Exception:
        traceback.print_exc()
        logger.error("Digitization: failed to build (y, value) pairs from axis data.")
        return [], [], [0], [], [], [], []

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
        return [], [], [0], [], [], [], []

    # End-point linear calibration: value = a * y + b
    a = (vals_axis[-1] - vals_axis[0]) / (ys_axis[-1] - ys_axis[0])
    b = vals_axis[0] - a * ys_axis[0]

    # Process curve coordinates: one point per column, as in the original implementation
    b_coords = top_curve_coords
    if b_coords is None or len(b_coords) == 0:
        logger.error("Digitization: top_curve_coords is empty - nothing to digitize.")
        return [], [], [0], [], [], [], []

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
        return [], [], [0], [], [], [], []

    # Normalize X to [0, 1] as arbitrary time axis
    Xmin_pix = min(X_pixels)
    Xmax_pix = max(X_pixels)
    if Xmax_pix == Xmin_pix:
        Xplot = [0.0 for _ in X_pixels]
    else:
        Xplot = [(x - Xmin_pix) / (Xmax_pix - Xmin_pix) for x in X_pixels]

    # Map pixel y to physical value using the axis calibration
    Yplot = [a * y + b for y in Y_pixels]

    Xplot_o, Yplot_o = [], []
    if overlay_curve_coords is not None and len(overlay_curve_coords) > 0:
        b_arr_o = [list(B) for B in overlay_curve_coords]
        b_swapped_o = [x[::-1] for x in b_arr_o]
        df_o = (
            pd.DataFrame(b_swapped_o)
            .groupby(0, as_index=False)[1]
            .mean()
            .values.tolist()
        )
        b_clean_o = [x[::-1] for x in df_o]
        Xm = [pt[1] for pt in b_clean_o]
        Ym = [pt[0] for pt in b_clean_o]
        if len(Xm) >= 2:
            if Xmax_pix > Xmin_pix:
                Xplot_o = [
                    (x - Xmin_pix) / (Xmax_pix - Xmin_pix) for x in Xm
                ]
            else:
                Xplot_o = [0.0 for _ in Xm]
            Yplot_o = [a * y + b for y in Ym]

    Xplot_g, Yplot_g = [], []
    if grow_curve_coords is not None and len(grow_curve_coords) > 0:
        b_arr_g = [list(B) for B in grow_curve_coords]
        b_swapped_g = [x[::-1] for x in b_arr_g]
        df_g = (
            pd.DataFrame(b_swapped_g)
            .groupby(0, as_index=False)[1]
            .mean()
            .values.tolist()
        )
        b_clean_g = [x[::-1] for x in df_g]
        Xm_g = [pt[1] for pt in b_clean_g]
        Ym_g = [pt[0] for pt in b_clean_g]
        if len(Xm_g) >= 2:
            if Xmax_pix > Xmin_pix:
                Xplot_g = [
                    (x - Xmin_pix) / (Xmax_pix - Xmin_pix) for x in Xm_g
                ]
            else:
                Xplot_g = [0.0 for _ in Xm_g]
            Yplot_g = [a * y + b for y in Ym_g]

    # Invert waveform if mean is negative (apply to overlay too)
    if np.mean(Yplot) < 0:
        Yplot = [y * (-1) for y in Yplot]
        Yplot_o = [y * (-1) for y in Yplot_o]
        Yplot_g = [y * (-1) for y in Yplot_g]

    # Additional smoothing specifically for the digitized ray series.
    # This reduces residual jitter visible in the ray-traced digitization plot.
    if overlay_is_ray:
        Yplot_o = _smooth_1d_digitized_shape_preserving(Yplot_o)
    else:
        Yplot = _smooth_1d_digitized_shape_preserving(Yplot)
    if len(Yplot_g) >= 2:
        Yplot_g = _smooth_1d_digitized_shape_preserving(Yplot_g)

    Ynought = [0.0]

    plt.figure(2)
    if len(Xplot_o) >= 2:
        if overlay_is_ray:
            plt.plot(Xplot, Yplot, "-", color=MORPH_CURVE_COLOR, linewidth=1.2, label="morph")
            plt.plot(Xplot_o, Yplot_o, "-", color="red", linewidth=1.2, label="ray")
        else:
            plt.plot(Xplot, Yplot, "-", color="red", linewidth=1.2, label="ray")
            plt.plot(Xplot_o, Yplot_o, "-", color=MORPH_CURVE_COLOR, linewidth=1.2, label="morph")
    elif len(Xplot_g) >= 2:
        plt.plot(Xplot, Yplot, "-", color="red", linewidth=1.2, label="ray")
    else:
        plt.plot(Xplot, Yplot, "-")
    if len(Xplot_g) >= 2:
        plt.plot(
            Xplot_g,
            Yplot_g,
            "-",
            color=GROW_CURVE_COLOR,
            linewidth=1.0,
            label="grow",
        )
    if len(Xplot_o) >= 2 or len(Xplot_g) >= 2:
        plt.legend(loc="best", fontsize=8)
    plt.xlabel("Arbitrary time scale")
    plt.ylabel("Flowrate (cm/s)")

    return Xplot, Yplot, Ynought, Xplot_o, Yplot_o, Xplot_g, Yplot_g


def _dicom_xy_from_curve_coords(top_curve_coords, dicom_metadata):
    """Physical (X, Y) lists from ``(row,col)`` curve and DICOM geometry."""
    if top_curve_coords is None or len(top_curve_coords) == 0:
        return [], []

    b_arr = [list(pt) for pt in top_curve_coords]
    b_swapped = [pt[::-1] for pt in b_arr]
    grouped = (
        pd.DataFrame(b_swapped).groupby(0, as_index=False)[1].mean().values.tolist()
    )
    b_clean = [pt[::-1] for pt in grouped]
    b_clean.sort(key=lambda pt: pt[1])

    col_pixels = np.asarray([pt[1] for pt in b_clean], dtype=float)
    row_pixels = np.asarray([pt[0] for pt in b_clean], dtype=float)

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

    Xplot = x_ref_phys + (col_pixels - x_ref) * dx
    Yplot = y_ref_phys + (row_pixels - y_ref) * dy
    return list(Xplot), list(Yplot)


def plot_digitized_data_dicom(
    dicom_metadata,
    top_curve_coords=None,
    overlay_curve_coords=None,
    overlay_is_ray=False,
    grow_curve_coords=None,
):
    """
    Digitize waveform for DICOM using metadata. Uses the same curve ordering as
    plot_digitized_data_single_axis: one point per column, sorted left-to-right,
    so plt.plot(Xplot, Yplot) draws a proper waveform.

    ``overlay_curve_coords`` / ``overlay_is_ray`` match
    ``plot_digitized_data_single_axis`` (blue morph, red ray).
    ``grow_curve_coords`` optional third series (violet).

    Returns:
        Xplot, Yplot, Ynought, Xplot_overlay, Yplot_overlay (overlay lists may be empty),
        Xplot_grow, Yplot_grow (grow lists may be empty).
    """
    Ynought = [float(dicom_metadata.get("ReferencePixelPhysicalValueY", 0.0))]

    if top_curve_coords is None or len(top_curve_coords) == 0:
        return [], [], Ynought, [], [], [], []

    Xplot, Yplot = _dicom_xy_from_curve_coords(top_curve_coords, dicom_metadata)

    Xplot_o, Yplot_o = [], []
    if overlay_curve_coords is not None and len(overlay_curve_coords) > 0:
        Xplot_o, Yplot_o = _dicom_xy_from_curve_coords(
            overlay_curve_coords, dicom_metadata
        )

    Xplot_g, Yplot_g = [], []
    if grow_curve_coords is not None and len(grow_curve_coords) > 0:
        Xplot_g, Yplot_g = _dicom_xy_from_curve_coords(
            grow_curve_coords, dicom_metadata
        )

    # Additional smoothing specifically for the digitized ray series.
    if overlay_is_ray:
        Yplot_o = _smooth_1d_digitized_shape_preserving(Yplot_o)
    else:
        Yplot = _smooth_1d_digitized_shape_preserving(Yplot)
    if len(Yplot_g) >= 2:
        Yplot_g = _smooth_1d_digitized_shape_preserving(Yplot_g)

    plt.figure(2)
    plt.clf()  # clear so each DICOM file gets a fresh plot (no accumulation from previous files)
    if len(Xplot_o) >= 2:
        if overlay_is_ray:
            plt.plot(Xplot, Yplot, "-", color=MORPH_CURVE_COLOR, linewidth=1.2, label="morph")
            plt.plot(Xplot_o, Yplot_o, "-", color="red", linewidth=1.2, label="ray")
        else:
            plt.plot(Xplot, Yplot, "-", color="red", linewidth=1.2, label="ray")
            plt.plot(Xplot_o, Yplot_o, "-", color=MORPH_CURVE_COLOR, linewidth=1.2, label="morph")
    elif len(Xplot_g) >= 2:
        plt.plot(Xplot, Yplot, "-", color="red", linewidth=1.2, label="ray")
    else:
        plt.plot(Xplot, Yplot, "-")
    if len(Xplot_g) >= 2:
        plt.plot(
            Xplot_g,
            Yplot_g,
            "-",
            color=GROW_CURVE_COLOR,
            linewidth=1.0,
            label="grow",
        )
    if len(Xplot_o) >= 2 or len(Xplot_g) >= 2:
        plt.legend(loc="best", fontsize=8)
    plt.xlabel("Physical X (time or distance)")
    plt.ylabel("Physical Y (e.g. velocity)")

    return Xplot, Yplot, Ynought, Xplot_o, Yplot_o, Xplot_g, Yplot_g


def waveform_metrics_from_digitized(
    Xplot,
    Yplot,
    Xplot_compare=None,
    Yplot_compare=None,
    Xplot_grow=None,
    Yplot_grow=None,
):
    """
    Compute waveform metrics from digitized x,y: Peak systolic (PS), End diastolic (ED),
    and metrics derived only from those: S/D, RI, TAmax, PI. Used for DICOM; returns a
    DataFrame with the same structure so downstream (Text_data, export, HTML) can use it.

    Derived (see ``usseg.hemodynamic_indices`` for RI, TAmax, PI):
      S/D = PS / ED; RI = (PS - ED) / PS; TAmax = temporal mean of ``y`` (envelope);
      PI = (PS - ED) / TAmax so PI and TAmax use the same denominator.

    Args:
        Xplot (list of float): X coordinates (time or physical axis).
        Yplot (list of float): Y coordinates (e.g. velocity).
        Xplot_compare, Yplot_compare: optional second curve (morph).
            Contract: primary (Xplot/Yplot) is ray.
        Xplot_grow, Yplot_grow: optional third curve (region grow).

    Returns:
        pandas.DataFrame: ``Digitized Value (ray)``, ``Digitized Value (morph)``, and
        ``Digitized Value (grow)`` when grow coordinates are present.
    """
    columns = [
        "Line",
        "Word",
        "Value",
        "Unit",
        "Digitized Value (ray)",
        "Digitized Value (morph)",
        "Digitized Value (grow)",
    ]
    empty_df = pd.DataFrame(columns=columns)

    if Xplot is None or Yplot is None or len(Xplot) == 0 or len(Yplot) == 0 or len(Xplot) != len(Yplot):
        return empty_df

    y = np.array(Yplot, dtype=float)
    if len(y) < 3:
        return empty_df

    try:
        x = np.array(Xplot, dtype=float)
        peaks_f, troughs_f, values = _waveform_peaks_troughs_values_from_physical_x(x, y)
        if values is None:
            return empty_df

        use_both = (
            Xplot_compare is not None
            and Yplot_compare is not None
            and len(Xplot_compare) >= 2
            and len(Xplot_compare) == len(Yplot_compare)
        )
        x2 = y2 = None
        peaks_f2 = np.array([], dtype=int)
        troughs_f2 = np.array([], dtype=int)
        values2 = None
        if use_both:
            x2 = np.asarray(Xplot_compare, dtype=float)
            y2 = np.asarray(Yplot_compare, dtype=float)
            peaks_f2, troughs_f2, values2 = _waveform_peaks_troughs_values_from_physical_x(x2, y2)

        use_grow = (
            Xplot_grow is not None
            and Yplot_grow is not None
            and len(Xplot_grow) >= 2
            and len(Xplot_grow) == len(Yplot_grow)
        )
        x3 = y3 = None
        peaks_f3 = np.array([], dtype=int)
        troughs_f3 = np.array([], dtype=int)
        values3 = None
        if use_grow:
            x3 = np.asarray(Xplot_grow, dtype=float)
            y3 = np.asarray(Yplot_grow, dtype=float)
            peaks_f3, troughs_f3, values3 = _waveform_peaks_troughs_values_from_physical_x(x3, y3)

        if len(x) == len(y):
            plt.figure(2)
            plt.clf()
            if use_both:
                plt.plot(x, y, "-", color="red", linewidth=1.2, label="ray")
                plt.plot(x2, y2, "-", color=MORPH_CURVE_COLOR, linewidth=1.2, label="morph")
            else:
                plt.plot(x, y, "-", color="red", linewidth=1.2, label="ray")
            if use_grow and x3 is not None and y3 is not None:
                plt.plot(x3, y3, "-", color=GROW_CURVE_COLOR, linewidth=1.0, label="grow")
            if use_both or use_grow:
                plt.legend(loc="best", fontsize=8)
            plt.xlabel("Physical X (time or distance)")
            plt.ylabel("Physical Y (e.g. velocity)")
            if use_both and values2 is not None:
                if len(peaks_f) > 0:
                    plt.plot(x[peaks_f], y[peaks_f], "x", color="C0", markersize=8, label="PS (ray)")
                if len(troughs_f) > 0:
                    plt.plot(x[troughs_f], y[troughs_f], "x", color="C1", markersize=8, label="ED (ray)")
                if len(peaks_f2) > 0:
                    plt.plot(x2[peaks_f2], y2[peaks_f2], "+", color=MORPH_CURVE_COLOR, markersize=8, label="PS (morph)")
                if len(troughs_f2) > 0:
                    plt.plot(x2[troughs_f2], y2[troughs_f2], "v", color=MORPH_CURVE_COLOR, markersize=8, label="ED (morph)")
            else:
                if len(peaks_f) > 0:
                    plt.plot(x[peaks_f], y[peaks_f], "x", color="C0", markersize=8, label="PS")
                if len(troughs_f) > 0:
                    plt.plot(x[troughs_f], y[troughs_f], "x", color="C1", markersize=8, label="ED")
            if use_grow and values3 is not None and x3 is not None and y3 is not None:
                if len(peaks_f3) > 0:
                    plt.plot(
                        x3[peaks_f3],
                        y3[peaks_f3],
                        "^",
                        color=GROW_CURVE_COLOR,
                        markersize=7,
                        label="PS (grow)",
                    )
                if len(troughs_f3) > 0:
                    plt.plot(
                        x3[troughs_f3],
                        y3[troughs_f3],
                        "v",
                        color=GROW_CURVE_COLOR,
                        markersize=7,
                        label="ED (grow)",
                    )

        words = ["PS", "ED", "S/D", "RI", "TA", "PI"]
        rows = []
        for i, (w, v) in enumerate(zip(words, values)):
            row = {
                "Line": i + 1,
                "Word": w,
                "Value": v,
                "Unit": "",
                "Digitized Value (ray)": "",
                "Digitized Value (morph)": "",
                "Digitized Value (grow)": "",
            }
            if use_both and values2 is not None:
                row["Digitized Value (ray)"] = v
                row["Digitized Value (morph)"] = values2[i]
            else:
                row["Digitized Value (ray)"] = v
            if use_grow and values3 is not None:
                row["Digitized Value (grow)"] = values3[i]
            rows.append(row)
        return pd.DataFrame(rows, columns=columns)
    except Exception:
        logger.warning("waveform_metrics_from_digitized failed", exc_info=True)
        return empty_df


def mean_wave(x_values, y_values, verbose=False):
    """
    Compute an average beat waveform from a contiguous Doppler waveform.

    Method summary (aligned with current usseg beat logic):
    1) Propose systolic anchor peaks with prominence-based peak finding and
       merge peaks that are too close.
    2) For each anchor peak, build a backward search window in the preceding
       part of the beat.
    3) In that window:
       - smooth the signal,
       - compute first derivative (slope) and second derivative (change in slope),
       - find a second-derivative anchor (strongest upslope acceleration),
       - walk backward on first derivative to the onset of low slope (foot onset).
    4) Build beats foot-to-foot and derive PS/ED points within each beat for
       diagnostics.
    5) Segment foot-to-foot, align beats to a common x-axis, average, remove
       outlier beats, and recompute the final mean wave.

    Parameters
    ----------
    x_values : numpy array
        Array of x values (typically time or sample position).
    y_values : numpy array
        Array of y values (waveform amplitude / velocity envelope).
    verbose : bool, optional
        If True, plot intermediate results.

    Returns
    -------
    new_average_wave : numpy array
        Average waveform (after beat-wise outlier filtering when applicable).
    x_common : numpy array
        Common x-axis corresponding to the average waveform.
    std_wave : numpy array
        Sample standard deviation across retained beats at each ``x_common`` point
        (same length as ``new_average_wave``). Variance beat-to-beat at phase *i* is
        ``std_wave[i] ** 2``.
    """
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    wave_amplitude = y_values.max()-y_values.min()

    peak_indices, _ = find_peaks(y_values, prominence=wave_amplitude / 4)
    # Min separation (assume x is time): 200 bpm -> 0.3 s; merge peaks closer than that, keep highest
    if len(peak_indices) > 1 and len(x_values) >= 2:
        dx = float(np.median(np.diff(x_values)))
        if np.isfinite(dx) and dx > 0:
            min_distance = max(1, int(60.0 / 200.0 / dx))
            order = np.argsort(peak_indices)
            peaks = peak_indices[order]
            consolidated = []
            i = 0
            while i < len(peaks):
                j = i
                best = int(peaks[i])
                while j + 1 < len(peaks) and (int(peaks[j + 1]) - int(peaks[j])) <= min_distance:
                    j += 1
                    cand = int(peaks[j])
                    if y_values[cand] > y_values[best]:
                        best = cand
                consolidated.append(best)
                i = j + 1
            peak_indices = np.array(consolidated, dtype=int)
    # ------------------------------------------------------------------
    # Foot finder aligned with current usseg logic:
    # second-derivative anchor -> backward first-derivative onset threshold.
    # ------------------------------------------------------------------
    foot_indices = []
    search_windows = []
    debug_rows = []

    search_fraction = 0.50
    smooth_window_max = 11
    polyorder = 2
    min_samples_before_peak = 3
    foot_max_rel_height = 0.55

    for i in range(0, len(peak_indices)):
        peak = int(peak_indices[i])
        if i == 0:
            if len(peak_indices) >= 3:
                diffs = np.diff(peak_indices).astype(float)
                other = diffs[1:] if len(diffs) >= 2 else diffs
                interval_est = int(np.round(np.mean(other))) if other.size > 0 else 0
            elif len(peak_indices) >= 2:
                interval_est = int(peak_indices[1] - peak_indices[0])
            else:
                interval_est = 0
            if interval_est < 5:
                continue
            interval = int(interval_est)
            prev_peak = max(0, peak - interval)
        else:
            prev_peak = int(peak_indices[i - 1])
            interval = int(peak - prev_peak)
        if interval < 5:
            continue

        local_search_fraction = float(search_fraction) if i == 0 else max(0.0, float(search_fraction) - 0.10)
        search_len = max(3, int(local_search_fraction * interval))
        # If first-wave search would extend before signal start, skip this beat.
        if i == 0 and (peak - search_len) < 0:
            continue
        search_start = max(prev_peak, peak - search_len)
        search_end = max(search_start + 2, peak - int(max(1, min_samples_before_peak)))
        if search_end <= search_start + 2:
            continue

        x_region_raw = np.asarray(x_values[search_start:search_end], dtype=float)
        y_region = y_values[search_start:search_end]
        if len(y_region) < 3 or x_region_raw.size != len(y_region):
            continue

        y_smooth = y_region.copy()
        if len(y_region) >= 5:
            win = min(smooth_window_max, len(y_region))
            if win % 2 == 0:
                win -= 1
            if win >= 5:
                y_smooth = savgol_filter(y_region, window_length=win, polyorder=polyorder)

        try:
            if np.all(np.isfinite(x_region_raw)) and (x_region_raw[-1] > x_region_raw[0]):
                x_region = np.linspace(float(x_region_raw[0]), float(x_region_raw[-1]), int(len(x_region_raw)))
                y_for_deriv = np.interp(x_region, x_region_raw, y_smooth)
            else:
                x_region = np.arange(search_end - search_start, dtype=float)
                y_for_deriv = y_smooth
        except Exception:
            x_region = np.arange(search_end - search_start, dtype=float)
            y_for_deriv = y_smooth

        dy = np.gradient(y_for_deriv, x_region)
        d2y_raw = np.gradient(dy, x_region)
        d2y = d2y_raw.copy()
        if len(y_region) >= 5:
            win_d = min(smooth_window_max, len(y_region))
            if win_d % 2 == 0:
                win_d -= 1
            if win_d >= 5:
                d2y = savgol_filter(d2y_raw, window_length=win_d, polyorder=polyorder)

        edge_guard = int(max(0, min(2, (len(d2y) - 1) // 2)))
        if len(d2y) - (2 * edge_guard) >= 3:
            d2_core = d2y[edge_guard: len(d2y) - edge_guard]
            foot2_local = int(edge_guard + np.argmax(d2_core))
        else:
            foot2_local = int(np.argmax(d2y))

        dy_seg = dy[: foot2_local + 1]
        if dy_seg.size == 0:
            continue
        dy_max = float(np.max(dy_seg))
        picked_local = int(foot2_local)
        slope_thr = np.nan
        if np.isfinite(dy_max) and dy_max > 0:
            slope_thr = 0.08 * dy_max
            for j in range(int(foot2_local), -1, -1):
                if float(dy[j]) <= float(slope_thr):
                    picked_local = int(j)
                    break
        picked = int(search_start + picked_local)

        try:
            trough_y = float(np.min(y_values[prev_peak:peak])) if peak > prev_peak + 1 else float(y_values[prev_peak])
            peak_y = float(y_values[peak])
        except Exception:
            trough_y = float(np.min(y_region))
            peak_y = float(np.max(y_region))
        allowed_y = trough_y + float(foot_max_rel_height) * (peak_y - trough_y)
        needs_fallback = (
            picked < 0
            or picked >= len(y_values)
            or picked >= peak - int(max(1, min_samples_before_peak))
            or float(y_values[picked]) > allowed_y
        )
        if needs_fallback:
            seg_pre = y_values[search_start: search_start + foot2_local + 1]
            if seg_pre.size > 0:
                picked = int(search_start + int(np.argmin(seg_pre)))
            else:
                picked = int(search_start + foot2_local)

        foot_indices.append(picked)
        search_windows.append((search_start, search_end))
        debug_rows.append(
            {
                "search_start": int(search_start),
                "search_end": int(search_end),
                "foot2_global": int(search_start + foot2_local),
                "picked_global": int(picked),
                "dy": np.asarray(dy, dtype=float),
                "d2y": np.asarray(d2y, dtype=float),
                "slope_thr": float(slope_thr) if np.isfinite(slope_thr) else np.nan,
            }
        )

    foot_indices = np.asarray(foot_indices, dtype=int)
    if len(foot_indices) < 2:
        raise ValueError("Not enough foot points found to calculate mean wave.")
    # PS/ED from foot-defined beats (same policy as usseg).
    y_s = np.asarray(y_values, dtype=float)
    if len(y_values) >= 5:
        y_s = np.convolve(y_values, np.ones(5) / 5.0, mode="same")
    ps_indices = []
    ed_indices = []
    for i in range(len(foot_indices) - 1):
        a = int(foot_indices[i])
        b = int(foot_indices[i + 1])
        if b <= a + 2:
            continue
        seg = y_s[a:b]
        anchor_in_beat = peak_indices[(peak_indices >= a) & (peak_indices < b)]
        ps_idx = None
        if anchor_in_beat.size > 0:
            aa = anchor_in_beat[np.argmax(y_s[anchor_in_beat])]
            ps_idx = int(aa)
            ps_indices.append(ps_idx)
        else:
            ps_idx = int(a + int(np.argmax(seg)))
            ps_indices.append(ps_idx)

        # ED is constrained to occur after PS within the same beat.
        ed_start = int(max(a, ps_idx + 1))
        if ed_start < b:
            seg_ed = y_s[ed_start:b]
            if seg_ed.size > 0:
                ed_indices.append(int(ed_start + int(np.argmin(seg_ed))))
                continue
        # Fallback for very short post-PS segments.
        ed_indices.append(int(a + int(np.argmin(seg))))
    segment_indices = foot_indices


    interpolated_waves = []
    if verbose:
        logger.warning("mean_wave(verbose=True): use scratch/mean_wave_test.py for diagnostic plots")

    for i in range(len(segment_indices) - 1):
        # Extract data for the current segment
        start_index = segment_indices[i]
        end_index = segment_indices[i + 1]
        x_segment = x_values[start_index:end_index]
        y_segment = y_values[start_index:end_index]

        # Shift x-coordinates for alignment (except the first wave)
        if i > 0:
            x_segment = x_segment - (x_segment[0] - x_values[segment_indices[0]])

        # Initialize the common x-axis using the first segment
        if i == 0:
            x_min = x_segment[0]
            x_max = x_segment[-1]
            x_common_points = len(x_segment)
            x_common = np.linspace(x_min, x_max, x_common_points)

        # Interpolate to the common x-axis
        interp_y = interp1d(x_segment, y_segment, kind='linear', fill_value="extrapolate")(x_common)
        interpolated_waves.append(interp_y)

    # Convert the list of interpolated waves to a NumPy array for calculations
    interpolated_waves_np = np.vstack(interpolated_waves)

    # Calculate the initial average and standard deviation
    average_wave = np.mean(interpolated_waves_np, axis=0)
    std_wave = np.std(interpolated_waves_np, axis=0)
    amplitude_of_ave = np.max(average_wave)-np.min(average_wave)

    if verbose:
        plt.figure(figsize=(10, 6))
        plt.title("Set of Waveforms")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        for wave_index, waveform in enumerate(interpolated_waves):
            plt.plot(x_common, waveform, label=f"Waveform {wave_index}")
        plt.plot(x_common, average_wave, label='Average wave', linestyle='-.')
        plt.plot(x_common, average_wave+std_wave, label='Average wave + SD', linestyle='--')
        plt.plot(x_common, average_wave-std_wave, label='Average wave - SD', linestyle='--')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Filter out waves outside the range of average ± standard deviation
    threshold_percentage = 80

    filtered_waves = []
    excluded_waves = []
    count_excluded = 0
    for wave in interpolated_waves_np:
        # Calculate the percentage of points that meet the OR condition
        within_range = (wave >= (average_wave - 0.2*amplitude_of_ave)) & (
                    wave <= (average_wave + 0.2*amplitude_of_ave))  # Points above or equal to lower bound
        percentage_within_range = np.sum(within_range) / len(wave) * 100
        # Include the wave if the percentage is above the threshold
        if percentage_within_range >= threshold_percentage:
            filtered_waves.append(wave)
        else:
            count_excluded += 1
            excluded_waves.append(wave)
    if verbose:
        print("Waves filtered, num excluded", count_excluded)

    if verbose:
        print(f"{len(interpolated_waves)} waveforms included in the calculationg for the average waveform,"
              f"using a cutoff proportion of {threshold_percentage} % for points within one standard deviation of the "
              f"raw native waveform")
    # Recalculate the average and standard deviation with the filtered waves
    if len(filtered_waves) == 0:
        filtered_waves_np = interpolated_waves_np
    else:
        filtered_waves_np = np.vstack(filtered_waves)
    new_average_wave = np.mean(filtered_waves_np, axis=0)
    new_std_wave = np.std(filtered_waves_np, axis=0)
    if verbose:
        plt.figure(figsize=(10, 6))
        plt.title("Average waveform")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.plot(x_common, new_average_wave)
        #plt.plot(x_common, excluded_waves[0])
        plt.grid(True)
        plt.show()

    return new_average_wave, x_common, new_std_wave


def save_mean_waves_ray_morph_grow_figure(
    out_path,
    x_ray,
    y_ray,
    x_morph,
    y_morph,
    x_grow,
    y_grow,
):
    """
    Single figure: mean beat waveform for ray, morph, and grow. Colours match the
    digitized overlay (red / MORPH_CURVE_COLOR / GROW_CURVE_COLOR). Each trace is
    shown on normalized beat phase [0, 1] so different beat lengths overlay.

    For each method, a translucent band shows **±1 sample standard deviation** across
    retained beats at each phase (variance = band half-width squared in (cm/s)²).
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    series = [
        (x_ray, y_ray, "ray", "red"),
        (x_morph, y_morph, "morph", MORPH_CURVE_COLOR),
        (x_grow, y_grow, "grow", GROW_CURVE_COLOR),
    ]
    n_plotted = 0
    for xv, yv, label, color in series:
        if xv is None or yv is None:
            continue
        xa = np.asarray(xv, dtype=float)
        ya = np.asarray(yv, dtype=float)
        if xa.size < 3 or xa.size != ya.size:
            continue
        try:
            y_avg, x_c, y_std = mean_wave(xa, ya, verbose=False)
        except Exception:
            logger.info("mean_wave failed for %s (need >=2 feet)", label)
            continue
        if x_c is None or y_avg is None or len(x_c) < 2 or len(y_avg) != len(x_c):
            continue
        span = float(x_c[-1] - x_c[0])
        if np.isfinite(span) and span > 0:
            xn = (np.asarray(x_c, dtype=float) - x_c[0]) / span
        else:
            xn = np.linspace(0.0, 1.0, len(x_c))
        y_avg = np.asarray(y_avg, dtype=float)
        y_std = np.nan_to_num(np.asarray(y_std, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if y_std.shape != y_avg.shape:
            y_std = np.zeros_like(y_avg)
        lo = y_avg - y_std
        hi = y_avg + y_std
        ax.fill_between(xn, lo, hi, color=color, alpha=0.22, linewidth=0, zorder=1)
        ax.plot(xn, y_avg, color=color, linewidth=1.8, label=label, zorder=2)
        n_plotted += 1
    ax.set_xlabel("Normalized beat phase")
    ax.set_ylabel("Flowrate (cm/s)")
    ax.set_title("Mean beat ±1 SD (ray / morph / grow)")
    ax.grid(True, alpha=0.3)
    if n_plotted:
        ax.legend(loc="best", fontsize=9, framealpha=0.92)
    else:
        ax.text(
            0.5,
            0.5,
            "No mean wave (need >=2 feet per trace)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    fig.savefig(out_path, dpi=900, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

def _beat_detection_pass(x, y, min_distance):
    """
    Single pass of beat detection: smooth y, find_peaks with distance and prominence,
    notch merge, ED as minimum between consecutive peaks. Returns (peaks, troughs) as int arrays,
    or (None, None) if detection fails. Caller supplies min_distance (samples) appropriate for
    x units (length-based for arbitrary x, time-based for time-scaled x).
    """
    if len(y) < 3 or len(x) != len(y):
        return None, None
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Smooth y to reduce digitisation jitter; detection uses y_s.
    y_s = pd.Series(y).rolling(window=5, center=True, min_periods=1).median().to_numpy()
    # X spacing: needed for merge window (samples per beat) in notch suppression.
    if len(x) >= 2:
        dx = float(np.median(np.diff(x)))
    else:
        dx = 1.0
    if not np.isfinite(dx) or dx <= 0:
        dx = 1.0

    # Prominence: ignore small bumps; 20% of 5–95% amplitude range.
    amp = float(np.percentile(y_s, 95) - np.percentile(y_s, 5))
    prom = 0.20 * amp
    if not np.isfinite(prom) or prom <= 0:
        prom = None

    # find_peaks with distance (min_distance from caller function) and prominence.
    peaks, _ = find_peaks(y_s, distance=min_distance, prominence=prom)
    if len(peaks) == 0:
        return None, None

    # Notch merge: multiple peaks in one beat → keep only the tallest per window.
    if len(peaks) >= 3:
        median_pp = float(np.median(np.diff(x[peaks])))
        merge_window_s = 0.45 * median_pp
        merge_window_samples = max(1, int(merge_window_s / dx))
    else:
        merge_window_samples = min_distance
    consolidated = []
    i = 0
    while i < len(peaks):
        j = i
        best = int(peaks[i])
        while j + 1 < len(peaks) and (int(peaks[j + 1]) - int(peaks[j])) <= merge_window_samples:
            j += 1
            cand = int(peaks[j])
            if y_s[cand] > y_s[best]:
                best = cand
        consolidated.append(best)
        i = j + 1
    peaks = np.array(consolidated, dtype=int)

    # ED = minimum between consecutive systolic peaks.
    trough_list = []
    for i in range(len(peaks) - 1):
        a, b = int(peaks[i]), int(peaks[i + 1])
        if b > a + 1:
            seg = y_s[a:b]
            trough_list.append(a + int(np.argmin(seg)))
    troughs = np.array(trough_list, dtype=int)
    if len(troughs) == 0:
        return None, None
    return peaks, troughs


def _detect_feet_indices_from_peaks(
    x,
    y,
    peak_indices,
    search_fraction=FOOT_SEARCH_FRACTION,
    smooth_window_max=FOOT_DERIV_SMOOTH_WINDOW_MAX,
    polyorder=FOOT_DERIV_SMOOTH_POLYORDER,
    foot_max_rel_height=FOOT_MAX_REL_HEIGHT,
    min_samples_before_peak=FOOT_MIN_SAMPLES_BEFORE_PEAK,
    return_debug=False,
):
    """
    Foot detection anchored on consecutive systolic peaks.

    For each peak from the 2nd onward:
      1) Search backwards within the last ``search_fraction`` of the previous
         peak-to-peak interval.
      2) Lightly smooth the window (Savitzky–Golay).
      3) Compute 2nd derivative; take its maximum as an upstroke anchor (foot_from_d2).
      4) Walk backward from that anchor using 1st derivative and pick the first
         low-slope point as foot onset (with amplitude/peak-distance guards).

    A height constraint is applied to avoid picking "feet" near the systolic peak:
      y_foot <= trough + foot_max_rel_height*(peak - trough).

    Returns a sorted, unique int array of foot indices.
    If ``return_debug`` is True, also returns a list of per-window debug records
    containing the exact search bounds and derivative traces used by detection.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    peaks = np.asarray(peak_indices, dtype=int)
    if peaks.size < 2 or len(x) != len(y):
        return np.array([], dtype=int)

    feet = []
    debug_rows = []
    for i in range(0, len(peaks)):
        peak = int(peaks[i])
        if i == 0:
            # First-beat interval estimate from the average of OTHER peak intervals.
            # This is more stable when the first observed gap is atypical.
            if len(peaks) >= 3:
                diffs = np.diff(peaks).astype(float)
                other = diffs[1:] if len(diffs) >= 2 else diffs
                interval_est = int(np.round(np.mean(other))) if other.size > 0 else 0
            elif len(peaks) >= 2:
                interval_est = int(peaks[1] - peaks[0])
            else:
                interval_est = 0
            if interval_est < 5:
                continue
            interval = int(interval_est)
            prev_peak = max(0, peak - interval)
        else:
            prev_peak = int(peaks[i - 1])
            interval = int(peak - prev_peak)
        if interval < 5:
            continue

        # Keep first-wave window as configured; tighten subsequent windows by 10%.
        local_search_fraction = float(search_fraction) if i == 0 else max(0.0, float(search_fraction) - 0.10)
        search_len = max(3, int(local_search_fraction * interval))
        # If first-wave search would extend before signal start, skip this beat.
        if i == 0 and (peak - search_len) < 0:
            continue
        search_start = max(prev_peak, peak - search_len)
        search_end = max(search_start + 2, peak - int(max(1, min_samples_before_peak)))
        if search_end <= search_start + 2:
            continue

        # Use local physical x and resample to a uniform x grid in this window.
        # This keeps derivatives tied to waveform geometry while avoiding unstable
        # gradients from irregular sampling intervals.
        x_region_raw = np.asarray(x[search_start:search_end], dtype=float)
        y_region = y[search_start:search_end]
        if y_region.size < 3:
            continue
        if x_region_raw.size != y_region.size:
            continue

        y_smooth = y_region.copy()
        if y_region.size >= 5:
            win = min(int(smooth_window_max), int(y_region.size))
            if win % 2 == 0:
                win -= 1
            if win >= 5:
                po = int(max(2, min(int(polyorder), win - 2)))
                try:
                    y_smooth = scipy.signal.savgol_filter(
                        y_region, window_length=win, polyorder=po, mode="interp"
                    )
                except Exception:
                    y_smooth = y_region

        # Uniform-in-x resample (same length) before derivatives.
        x_region = x_region_raw
        y_for_deriv = y_smooth
        try:
            if np.all(np.isfinite(x_region_raw)) and (x_region_raw[-1] > x_region_raw[0]):
                x_region = np.linspace(float(x_region_raw[0]), float(x_region_raw[-1]), int(len(x_region_raw)))
                y_for_deriv = np.interp(x_region, x_region_raw, y_smooth)
            else:
                x_region = np.arange(search_end - search_start, dtype=float)
                y_for_deriv = y_smooth
        except Exception:
            x_region = np.arange(search_end - search_start, dtype=float)
            y_for_deriv = y_smooth

        try:
            dy = np.gradient(y_for_deriv, x_region)
            d2y_raw = np.gradient(dy, x_region)
            d3y_raw = np.gradient(d2y_raw, x_region)
        except Exception:
            continue

        # Additional derivative-stage smoothing for stability.
        d2y = d2y_raw
        d3y = d3y_raw
        if y_region.size >= 5:
            win_d = min(int(smooth_window_max), int(y_region.size))
            if win_d % 2 == 0:
                win_d -= 1
            if win_d >= 5:
                po_d = int(max(2, min(int(polyorder), win_d - 2)))
                try:
                    d2y = scipy.signal.savgol_filter(
                        d2y_raw, window_length=win_d, polyorder=po_d, mode="interp"
                    )
                except Exception:
                    d2y = d2y_raw
                try:
                    d3y = scipy.signal.savgol_filter(
                        d3y_raw, window_length=win_d, polyorder=po_d, mode="interp"
                    )
                except Exception:
                    d3y = d3y_raw

        # Upper bound from 2nd-derivative peak (interior-only to reduce edge artifacts).
        edge_guard = int(max(0, min(FOOT_DERIV_EDGE_GUARD, (len(d2y) - 1) // 2)))
        if len(d2y) - (2 * edge_guard) >= 3:
            d2_core = d2y[edge_guard : len(d2y) - edge_guard]
            foot2_local = int(edge_guard + np.argmax(d2_core))
        else:
            foot2_local = int(np.argmax(d2y))
        foot2_local = max(0, min(foot2_local, int(len(d2y) - 1)))

        # Candidate region for final foot: [search_start, search_start + foot2_local]
        if foot2_local < 1:
            continue

        # Reject "feet" that sit too close to the systolic peak amplitude.
        # Compute local trough/peak amplitude in the prev_peak->peak interval.
        try:
            trough_y = float(np.min(y[prev_peak:peak])) if peak > prev_peak + 1 else float(y[prev_peak])
            peak_y = float(y[peak])
        except Exception:
            trough_y = float(np.min(y_region))
            peak_y = float(np.max(y_region))

        allowed_y = trough_y + float(foot_max_rel_height) * (peak_y - trough_y)

        # Refine backwards from 2nd-derivative anchor using first derivative:
        # find onset of the upslope as the first low-slope point moving backward.
        dy_seg = dy[: foot2_local + 1]
        if dy_seg.size == 0:
            continue
        dy_max = float(np.max(dy_seg))
        picked_local = int(foot2_local)
        if np.isfinite(dy_max) and dy_max > 0:
            slope_thr = 0.08 * dy_max  # Tunable onset threshold (typical: 0.05-0.15)
            for j in range(int(foot2_local), -1, -1):
                if float(dy[j]) <= float(slope_thr):
                    picked_local = int(j)
                    break
        picked = int(search_start + picked_local)

        # Keep guards to avoid physiologically implausible picks.
        needs_fallback = (
            picked < 0
            or picked >= len(y)
            or picked >= peak - int(max(1, min_samples_before_peak))
            or float(y[picked]) > allowed_y
        )
        if needs_fallback:
            # Prefer a local trough before foot2 anchor; if none, fallback to foot2.
            seg_pre = y[search_start : search_start + foot2_local + 1]
            if seg_pre.size > 0:
                picked = int(search_start + int(np.argmin(seg_pre)))
            else:
                picked = int(search_start + foot2_local)
        feet.append(int(picked))
        if return_debug:
            debug_rows.append(
                {
                    "search_start": int(search_start),
                    "search_end": int(search_end),
                    "peak_index": int(peak),
                    "foot2_global": int(search_start + foot2_local),
                    "picked_global": int(picked),
                    "picked_local": int(max(0, picked - search_start)),
                    "x_region": np.asarray(x_region, dtype=float),
                    "dy": np.asarray(dy, dtype=float),
                    "d2y": np.asarray(d2y, dtype=float),
                    "d3y": np.asarray(d3y, dtype=float),
                    "slope_thr": float(0.08 * dy_max) if np.isfinite(dy_max) and dy_max > 0 else np.nan,
                    "allowed_y": float(allowed_y),
                }
            )

    if len(feet) == 0:
        if return_debug:
            return np.array([], dtype=int), debug_rows
        return np.array([], dtype=int)
    feet = np.array(sorted(set(int(i) for i in feet)), dtype=int)
    # Keep within valid range
    feet = feet[(feet >= 0) & (feet < len(y))]
    if return_debug:
        return feet, debug_rows
    return feet


def _peaks_troughs_from_feet(x, y, foot_indices, anchor_peaks=None):
    """
    Compute PS (peak) and ED (trough) indices per beat using foot-to-foot boundaries.

    Beat i spans [feet[i], feet[i+1]). Within each beat:
      PS index = argmax(y_s)
      ED index = argmin(y_s)

    Uses a light rolling-median smoothing for stability (matches existing logic).
    Returns (peaks, troughs) int arrays, each length n_beats.

    If ``anchor_peaks`` is provided, PS per beat prefers an anchor peak inside
    [feet[i], feet[i+1]); falls back to argmax(y_s) in that beat if none exists.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    feet = np.asarray(foot_indices, dtype=int)
    anchors = (
        np.asarray(anchor_peaks, dtype=int)
        if anchor_peaks is not None
        else np.array([], dtype=int)
    )
    if feet.size < 2 or len(x) != len(y):
        return np.array([], dtype=int), np.array([], dtype=int)

    # Stabilize jitter a bit for extremum selection inside beats.
    y_s = pd.Series(y).rolling(window=5, center=True, min_periods=1).median().to_numpy()

    peaks = []
    troughs = []
    for i in range(len(feet) - 1):
        a = int(feet[i])
        b = int(feet[i + 1])
        if b <= a + 2:
            continue
        seg = y_s[a:b]
        anchor_in_beat = anchors[(anchors >= a) & (anchors < b)]
        ps_idx = None
        if anchor_in_beat.size > 0:
            # Choose tallest anchor within this beat (closest to PS definition).
            aa = anchor_in_beat[np.argmax(y_s[anchor_in_beat])]
            ps_idx = int(aa)
            peaks.append(ps_idx)
        else:
            ps_idx = int(a + int(np.argmax(seg)))
            peaks.append(ps_idx)

        # ED should be after PS within the beat. If post-PS segment is too short,
        # fall back to beat-wide minimum.
        ed_start = int(max(a, ps_idx + 1))
        if ed_start < b:
            seg_ed = y_s[ed_start:b]
            if seg_ed.size > 0:
                troughs.append(int(ed_start + int(np.argmin(seg_ed))))
                continue
        troughs.append(int(a + int(np.argmin(seg))))

    return np.asarray(peaks, dtype=int), np.asarray(troughs, dtype=int)


def _sqi_template_correlation_from_feet(y, feet, template_len=200, min_corr=0.85):
    """
    SQI variant for foot-to-foot segmentation.

    Each beat is feet[i] -> feet[i+1]. Beats are resampled to template_len points,
    median template is computed, and beats with corr < min_corr are rejected.

    Returns:
        good_beats_mask (np.ndarray bool): length n_beats
    """
    y = np.asarray(y, dtype=float)
    feet = np.asarray(feet, dtype=int)
    if feet.size < 2:
        return np.zeros(0, dtype=bool)

    n_beats = int(feet.size - 1)
    resampled = np.zeros((n_beats, template_len), dtype=float)
    valid = np.ones(n_beats, dtype=bool)

    for i in range(n_beats):
        a = int(feet[i])
        b = int(feet[i + 1])
        if b <= a + 1:
            valid[i] = False
            resampled[i, :] = np.nan
            continue
        seg = y[a : b + 1]
        if seg.size < 3 or np.std(seg) < 1e-10:
            valid[i] = False
            resampled[i, :] = np.nan
            continue
        x_old = np.linspace(0, 1, len(seg))
        x_new = np.linspace(0, 1, template_len)
        resampled[i, :] = np.interp(x_new, x_old, seg)

    if np.sum(valid) == 0:
        return np.ones(n_beats, dtype=bool)

    template = np.nanmedian(resampled[valid, :], axis=0)
    if np.std(template) < 1e-10:
        return np.ones(n_beats, dtype=bool)

    good = np.zeros(n_beats, dtype=bool)
    for i in range(n_beats):
        if not valid[i]:
            good[i] = False
            continue
        r = resampled[i, :]
        c = np.corrcoef(r, template)[0, 1]
        good[i] = (c >= min_corr) if np.isfinite(c) else False

    # If all rejected, keep all (fallback).
    if not np.any(good):
        return np.ones(n_beats, dtype=bool)
    return good


def _beat_detection_pass_feet(x, y, min_distance):
    """
    Beat detection returning feet + PS/ED indices.

    Steps:
      1) Detect PS peaks (mean_wave-style: prominence ~ amplitude/4) with min_distance.
      2) Detect feet (mean_wave 3rd-derivative method) anchored on consecutive peaks.
      3) Compute PS (max) and ED (min) within each foot-to-foot beat.

    Returns:
      feet (int array), peaks (int array), troughs (int array)
    """
    # Peak proposing for feet: match mean_wave style (prominence = amplitude/4),
    # while respecting the caller-provided min_distance.
    try:
        y_arr = np.asarray(y, dtype=float)
        amp = float(np.max(y_arr) - np.min(y_arr))
        prom = amp / 4.0 if np.isfinite(amp) and amp > 0 else None
        peaks, _ = find_peaks(y_arr, distance=int(min_distance), prominence=prom)
    except Exception:
        peaks, _troughs_unused = _beat_detection_pass(x, y, min_distance)
    if peaks is None or len(peaks) == 0:
        logger.info(
            "beat_detect_feet: no peaks found (n=%d, min_distance=%d)",
            int(len(y)) if y is not None else -1,
            int(min_distance),
        )
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)

    feet = _detect_feet_indices_from_peaks(x, y, peaks)
    if feet.size < 2:
        logger.info(
            "beat_detect_feet: insufficient feet (peaks=%d feet=%d, n=%d)",
            int(len(peaks)),
            int(len(feet)),
            int(len(y)) if y is not None else -1,
        )
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)

    peaks_b, troughs_b = _peaks_troughs_from_feet(x, y, feet, anchor_peaks=peaks)
    if peaks_b.size == 0 or troughs_b.size == 0:
        logger.info(
            "beat_detect_feet: no beat extrema from feet (feet=%d peaks=%d troughs=%d, n=%d)",
            int(len(feet)),
            int(len(peaks_b)),
            int(len(troughs_b)),
            int(len(y)) if y is not None else -1,
        )
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
    return feet, peaks_b, troughs_b


def _digitized_peaks_metrics_timescaled(x, y, hr, arbitrary_period_primary):
    """
    Beat detection + SQI + PS/ED-based metrics for a digitized series on arbitrary x [0, 1],
    using the primary series' mean beat period for HR time scaling (matches plot overlay x scale).
    Returns (peaks_for_metrics, troughs_for_metrics, values_or_none) where values_or_none lists
    six rounded floats: PS, ED, S/D, RI, TAmax, PI. TAmax is the temporal mean of ``y``;
    PI uses that same mean (see ``usseg.hemodynamic_indices``).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    peaks_for_metrics = np.array([], dtype=int)
    troughs_for_metrics = np.array([], dtype=int)
    if len(y) < 3 or len(x) != len(y):
        return peaks_for_metrics, troughs_for_metrics, None
    try:
        try:
            hr = float(hr)
        except Exception:
            hr = 0.0
        if not np.isfinite(arbitrary_period_primary) or arbitrary_period_primary <= 0:
            arbitrary_period_primary = 1.0

        min_distance_pass1 = max(1, len(x) // 15)
        feet, peaks, troughs = _beat_detection_pass_feet(x, y, min_distance_pass1)

        if np.isfinite(hr) and hr > 0.0:
            real_period = 60.0 / hr
            scale_factor = real_period / arbitrary_period_primary
            x_time = x * scale_factor
        else:
            x_time = x.copy()

        # Indices do not change with linear time scaling, but we keep the second pass
        # for consistency with existing logic (distance in samples depends on dx).
        if x_time is not None and len(x_time) == len(y) and np.any(x_time != x):
            dx_s = float(np.median(np.diff(x_time))) if len(x_time) >= 2 else 1.0
            if np.isfinite(dx_s) and dx_s > 0:
                min_sep_s = 60.0 / 200.0
                min_distance_pass2 = max(1, int(min_sep_s / dx_s))
                feet2, peaks2, troughs2 = _beat_detection_pass_feet(
                    x_time, y, min_distance_pass2
                )
                if feet2.size >= 2 and peaks2.size > 0 and troughs2.size > 0:
                    feet, peaks, troughs = feet2, peaks2, troughs2

        peaks_for_metrics = peaks
        troughs_for_metrics = troughs
        if USE_SQI_FILTER and feet.size >= 2 and peaks.size > 0 and troughs.size > 0:
            good_beats = _sqi_template_correlation_from_feet(
                y, feet, template_len=200, min_corr=0.85
            )
            # Map beat mask -> peak/trough masks (one each per beat)
            n = min(len(good_beats), len(peaks), len(troughs))
            if n > 0 and np.any(good_beats[:n]):
                peaks_for_metrics = peaks[:n][good_beats[:n]]
                troughs_for_metrics = troughs[:n][good_beats[:n]]

        if len(peaks_for_metrics) > 0 and len(troughs_for_metrics) > 0:
            PS = float(statistics.mean(y[peaks_for_metrics]))
            ED = float(statistics.mean(y[troughs_for_metrics]))
            if ED == 0:
                ED = np.finfo(float).eps
            SoverD = PS / ED
            RI = resistive_index_from_ps_ed(PS, ED)
            mean_v = float(np.mean(y))
            TAmax = tamax_from_envelope_temporal_mean(mean_v)
            PI = pulsatility_index_from_ps_ed_and_mean_velocity(PS, ED, mean_v)
            values = [
                round(PS, 2),
                round(ED, 2),
                round(SoverD, 2),
                round(RI, 2),
                round(TAmax, 2),
                round(PI, 2),
            ]
            return peaks_for_metrics, troughs_for_metrics, values
    except Exception:
        pass
    return peaks_for_metrics, troughs_for_metrics, None


def _waveform_peaks_troughs_values_from_physical_x(x, y):
    """
    Beat detection + SQI + waveform metrics when x is already in physical units (e.g. DICOM time).
    Returns (peaks_f, troughs_f, values_or_none) where values_or_none is six rounded floats
    (PS, ED, S/D, RI, TAmax, PI) or None. TAmax is the temporal mean of ``y``; PI divides by the same mean.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    empty_p = np.array([], dtype=int)
    if len(y) < 3 or len(x) != len(y):
        return empty_p, empty_p, None
    try:
        y_s = pd.Series(y).rolling(window=5, center=True, min_periods=1).median().to_numpy()
        if len(x) >= 2:
            dx = float(np.median(np.diff(x)))
        else:
            dx = 1.0
        if not np.isfinite(dx) or dx <= 0:
            dx = 1.0
        min_sep_s = 60.0 / 200.0
        min_distance = max(1, int(min_sep_s / dx))
        amp = float(np.percentile(y_s, 95) - np.percentile(y_s, 5))
        prom = 0.20 * amp
        if not np.isfinite(prom) or prom <= 0:
            prom = None
        feet, peaks, troughs = _beat_detection_pass_feet(x, y, min_distance)
        if feet.size < 2 or peaks.size == 0 or troughs.size == 0:
            return empty_p, empty_p, None

        peaks_f = peaks
        troughs_f = troughs
        if USE_SQI_FILTER:
            good_beats = _sqi_template_correlation_from_feet(
                y, feet, template_len=200, min_corr=0.85
            )
            n = min(len(good_beats), len(peaks), len(troughs))
            if n > 0 and np.any(good_beats[:n]):
                peaks_f = peaks[:n][good_beats[:n]]
                troughs_f = troughs[:n][good_beats[:n]]
        PS = float(statistics.mean(y[peaks_f]))
        ED = float(statistics.mean(y[troughs_f]))
        if ED == 0:
            ED = np.finfo(float).eps
        SoverD = PS / ED
        RI = resistive_index_from_ps_ed(PS, ED)
        mean_v = float(np.mean(y))
        TAmax = tamax_from_envelope_temporal_mean(mean_v)
        PI = pulsatility_index_from_ps_ed_and_mean_velocity(PS, ED, mean_v)
        values = [
            round(PS, 2),
            round(ED, 2),
            round(SoverD, 2),
            round(RI, 2),
            round(TAmax, 2),
            round(PI, 2),
        ]
        return peaks_f, troughs_f, values
    except Exception:
        return empty_p, empty_p, None


def _sqi_template_correlation(y, peaks, troughs, template_len=200, min_corr=0.85):
    """
    SQI: cycle shape consistency via template correlation. Each beat is defined as
    peak-to-peak (segment from peaks[i] to peaks[i+1]). Beats are resampled to a
    fixed length, median template is built, and each beat is correlated to the template;
    beats below min_corr are rejected. With one beat, template equals that beat (corr=1).

    Returns:
        good_peaks_mask (np.ndarray bool): length len(peaks); True = keep for metrics/plot.
        good_troughs_mask (np.ndarray bool): length len(troughs); True = keep.
    If SQI cannot be applied (e.g. too few points), returns all True (no filtering).
    """
    y = np.asarray(y, dtype=float)
    peaks = np.asarray(peaks, dtype=int)
    troughs = np.asarray(troughs, dtype=int)
    n_peaks = len(peaks)
    n_troughs = len(troughs)
    # Expect one trough between each pair of consecutive peaks.
    if n_peaks < 2 or n_troughs != n_peaks - 1:
        return np.ones(n_peaks, dtype=bool), np.ones(n_troughs, dtype=bool)

    n_beats = n_peaks - 1
    # Resample each beat (peak[i] -> peak[i+1]) to template_len points.
    resampled = np.zeros((n_beats, template_len))
    for i in range(n_beats):
        a, b = int(peaks[i]), int(peaks[i + 1])
        if b <= a + 1:
            resampled[i, :] = np.nan
            continue
        seg = y[a : b + 1]
        x_old = np.linspace(0, 1, len(seg))
        x_new = np.linspace(0, 1, template_len)
        resampled[i, :] = np.interp(x_new, x_old, seg)

    # Drop beats that are all NaN or constant (would break correlation).
    valid = np.ones(n_beats, dtype=bool)
    for i in range(n_beats):
        r = resampled[i, :]
        if np.any(np.isnan(r)) or np.std(r) < 1e-10:
            valid[i] = False
    if np.sum(valid) == 0:
        return np.ones(n_peaks, dtype=bool), np.ones(n_troughs, dtype=bool)

    # Median template (over valid beats only).
    template = np.nanmedian(resampled[valid, :], axis=0)
    if np.std(template) < 1e-10:
        return np.ones(n_peaks, dtype=bool), np.ones(n_troughs, dtype=bool)

    # Pearson correlation of each (valid) beat to template.
    good_beat = np.zeros(n_beats, dtype=bool)
    for i in range(n_beats):
        if not valid[i]:
            good_beat[i] = False
            continue
        r = resampled[i, :]
        c = np.corrcoef(r, template)[0, 1]
        good_beat[i] = c >= min_corr if np.isfinite(c) else False

    # If all beats rejected, keep all (fallback: no filtering).
    if not np.any(good_beat):
        return np.ones(n_peaks, dtype=bool), np.ones(n_troughs, dtype=bool)

    # Map beat quality to peak/trough masks. Peak i is in beat i-1 (start) and beat i (end); trough i is in beat i only.
    good_peaks_mask = np.zeros(n_peaks, dtype=bool)
    good_peaks_mask[0] = good_beat[0]
    good_peaks_mask[-1] = good_beat[-1]
    for i in range(1, n_peaks - 1):
        good_peaks_mask[i] = good_beat[i - 1] or good_beat[i]
    good_troughs_mask = good_beat.copy()
    return good_peaks_mask, good_troughs_mask


def digitized_hr_scale_factor_for_raster(Xplot_ray, Yplot_ray, df):
    """
    Same HR + mean foot spacing rule as ``plot_correction``: arbitrary x (typically
    [0, 1] from ``plot_digitized_data_single_axis``) is multiplied by this factor
    to get time in seconds when OCR ``df`` contains a valid HR.

    Returns:
        scale_factor (float): multiply arbitrary x by this; 1.0 if HR missing/invalid
            or beat detection fails to yield foot spacing.
        hr (float): HR from df, or 0.0 if missing/invalid.
        arbitrary_period (float): mean x-spacing between feet on arbitrary axis, or 1.0.
    """
    x = np.asarray(Xplot_ray, dtype=float)
    y = np.asarray(Yplot_ray, dtype=float)
    arbitrary_period = 1.0
    if len(y) >= 3 and len(x) == len(y):
        min_distance_pass1 = max(1, len(x) // 15)
        feet1, peaks1, troughs1 = _beat_detection_pass_feet(x, y, min_distance_pass1)
        if feet1.size >= 2 and peaks1.size > 0 and troughs1.size > 0:
            feet = feet1
            if len(feet) >= 2:
                arbitrary_period = float(x[int(feet[-1])] - x[int(feet[0])]) / max(
                    1, len(feet) - 1
                )
    hr = 0.0
    try:
        hr_vals = df.loc[df["Word"].str.contains("HR"), "Value"].values
        hr = float(hr_vals[0]) if len(hr_vals) > 0 else 0.0
    except Exception:
        pass
    if not np.isfinite(hr) or hr <= 0.0 or not np.isfinite(arbitrary_period) or arbitrary_period <= 0:
        return 1.0, float(hr) if np.isfinite(hr) else 0.0, arbitrary_period
    real_period = 60.0 / hr
    scale_factor = real_period / arbitrary_period
    return float(scale_factor), float(hr), float(arbitrary_period)


def scale_raster_digitized_x(x_list, scale_factor):
    """Scale a digitized x series by ``scale_factor``; pass-through if empty or factor is 1."""
    if x_list is None or len(x_list) == 0 or scale_factor == 1.0:
        return x_list
    return (np.asarray(x_list, dtype=float) * float(scale_factor)).tolist()


def finalize_image_digitized_model_choice(
    df,
    x_ray,
    y_ray,
    x_morph,
    y_morph,
    x_grow,
    y_grow,
):
    """
    For **raster images** only: score ray / morph / grow digitized columns vs OCR ``Value``
    (same traffic-light rules as HTML), set ``df[\"Returned model\"]``, and return the
    selected ``(x, y)`` after HR scaling.

    Call after ``plot_correction`` (when digitized columns exist) and after any HR
    x-axis scaling so returned coordinates match the final time axis.

    If the chosen curve is too short, falls back to the first non-empty series among
    ray, morph, grow and updates ``Returned model`` accordingly.
    """
    pairs = {
        "ray": (list(x_ray or []), list(y_ray or [])),
        "morph": (list(x_morph or []), list(y_morph or [])),
        "grow": (list(x_grow or []), list(y_grow or [])),
    }

    def _pick_nonempty(preferred_key: str) -> tuple[str, list, list]:
        order = [preferred_key, "ray", "morph", "grow"]
        seen = set()
        for k in order:
            if k in seen:
                continue
            seen.add(k)
            xk, yk = pairs[k]
            if len(xk) >= 2 and len(xk) == len(yk):
                return k, xk, yk
        xr, yr = pairs["ray"]
        return "ray", xr, yr

    if df is None:
        _, xs, ys = _pick_nonempty("ray")
        return xs, ys

    if "Digitized Value (ray)" not in df.columns:
        best = "ray"
    else:
        best = select_best_digitized_model_for_image(df)

    k, x_pick, y_pick = _pick_nonempty(best)
    df["Returned model"] = returned_model_cell_value(k)
    return x_pick, y_pick


def plot_correction(
    Xplot,
    Yplot,
    df,
    Xplot_compare=None,
    Yplot_compare=None,
    Xplot_grow=None,
    Yplot_grow=None,
):
    """
    Adjusts and corrects the digitized waveform data using extracted text data, identifies
    and filters peaks and troughs, computes hemodynamic parameters, scales the time axis,
    and plots the corrected waveform.

    Two-pass beat detection: first on arbitrary x (to get period for scaling), then after
    time-scaling with HR the second pass uses time-based spacing. Metrics use the second
    pass when available, else the first.

    Args:
        Xplot (list of float): The x-coordinates (time axis) of the waveform data.
        Yplot (list of float): The y-coordinates (flowrate axis) of the waveform data.
        df (pandas.DataFrame): The DataFrame with extracted text data including 
                               hemodynamic parameters and heart rate.
        Xplot_compare, Yplot_compare: optional second digitized series (morph).
            Contract: primary (Xplot/Yplot) is ray, compare is morph.
        Xplot_grow, Yplot_grow: optional third digitized series (region grow).

    Returns:
        **df** (pandas.DataFrame): ``Digitized Value (ray)``, ``Digitized Value (morph)``,
        and ``Digitized Value (grow)`` when grow coordinates are present.
    """
    y = np.array(Yplot, dtype=float)
    x = np.array(Xplot, dtype=float)
    df.insert(loc=3, column="Digitized Value (ray)", value="")
    df.insert(loc=4, column="Digitized Value (morph)", value="")
    df.insert(loc=5, column="Digitized Value (grow)", value="")
    peaks = np.array([], dtype=int)
    troughs = np.array([], dtype=int)
    peaks_for_metrics = np.array([], dtype=int)
    troughs_for_metrics = np.array([], dtype=int)
    peaks_compare_m = np.array([], dtype=int)
    troughs_compare_m = np.array([], dtype=int)
    peaks_grow_m = np.array([], dtype=int)
    troughs_grow_m = np.array([], dtype=int)
    arbitrary_period = 1.0
    x_time = None
    hr = 0.0

    try:
        # First pass: beat detection on arbitrary x [0, 1]. Length-based min distance
        # (assume at most ~15 beats in strip). Gives peaks/troughs and mean period for scaling.
        # -------------------------------------------------------------------------
        feet = np.array([], dtype=int)
        if len(y) >= 3 and len(x) == len(y):
            min_distance_pass1 = max(1, len(x) // 15)
            feet1, peaks1, troughs1 = _beat_detection_pass_feet(x, y, min_distance_pass1)
            if feet1.size >= 2 and peaks1.size > 0 and troughs1.size > 0:
                feet = feet1
                peaks = peaks1
                troughs = troughs1
                if len(feet) >= 2:
                    arbitrary_period = float(x[feet[-1]] - x[feet[0]]) / max(1, len(feet) - 1)
                else:
                    arbitrary_period = 1.0

        # -------------------------------------------------------------------------
        # Time scaling: get HR from df and scale x -> x_time (seconds). If HR is
        # missing or invalid we skip scaling and keep arbitrary x for plotting.
        # -------------------------------------------------------------------------
        try:
            hr_vals = df.loc[df["Word"].str.contains("HR"), "Value"].values
            hr = float(hr_vals[0]) if len(hr_vals) > 0 else 0.0
            if np.isfinite(hr) and hr > 0.0:
                real_period = 60.0 / hr
                scale_factor = real_period / arbitrary_period
                x_time = x * scale_factor
            else:
                x_time = x.copy()
        except Exception:
            x_time = x.copy()

        # Second pass: beat detection on time-scaled x when available. Time-based min
        # distance (200 bpm). If this succeeds we use these peaks/troughs; else keep first pass.
        # -------------------------------------------------------------------------
        if x_time is not None and len(x_time) == len(y) and np.any(x_time != x):
            dx_s = float(np.median(np.diff(x_time))) if len(x_time) >= 2 else 1.0
            if np.isfinite(dx_s) and dx_s > 0:
                min_sep_s = 60.0 / 200.0
                min_distance_pass2 = max(1, int(min_sep_s / dx_s))
                feet2, peaks2, troughs2 = _beat_detection_pass_feet(
                    x_time, y, min_distance_pass2
                )
                if feet2.size >= 2 and peaks2.size > 0 and troughs2.size > 0:
                    feet = feet2
                    peaks = peaks2
                    troughs = troughs2

        # SQI: keep only beats that pass template-correlation filter. Metrics and plot
        # use these peaks/troughs only.
        # -------------------------------------------------------------------------
        peaks_for_metrics = peaks
        troughs_for_metrics = troughs
        if USE_SQI_FILTER and feet.size >= 2 and len(peaks) > 0 and len(troughs) > 0:
            good_beats = _sqi_template_correlation_from_feet(
                y, feet, template_len=200, min_corr=0.85
            )
            n = min(len(good_beats), len(peaks), len(troughs))
            if n > 0 and np.any(good_beats[:n]):
                peaks_for_metrics = peaks[:n][good_beats[:n]]
                troughs_for_metrics = troughs[:n][good_beats[:n]]

        # Metrics from filtered peaks/troughs. Use original y; guard ED for S/D and RI.
        # -------------------------------------------------------------------------
        if len(peaks_for_metrics) > 0 and len(troughs_for_metrics) > 0:
            PS = float(statistics.mean(y[peaks_for_metrics]))
            ED = float(statistics.mean(y[troughs_for_metrics]))
            if ED == 0:
                ED = np.finfo(float).eps
            SoverD = PS / ED
            RI = resistive_index_from_ps_ed(PS, ED)
            mean_v = float(np.mean(y))
            TAmax = tamax_from_envelope_temporal_mean(mean_v)
            PI = pulsatility_index_from_ps_ed_and_mean_velocity(PS, ED, mean_v)
            if _is_ophthalmic_df(df):
                words = ["PS", "ED", "PS/ED", "RI", "PI"]
                values = [
                    round(PS, 2),
                    round(ED, 2),
                    round(SoverD, 2),
                    round(RI, 2),
                    round(PI, 2),
                ]
            else:
                words = ["PS", "ED", "S/D", "RI", "TA", "PI"]
                values = [
                    round(PS, 2),
                    round(ED, 2),
                    round(SoverD, 2),
                    round(RI, 2),
                    round(TAmax, 2),
                    round(PI, 2),
                ]
            has_compare = (
                Xplot_compare is not None
                and Yplot_compare is not None
                and len(Xplot_compare) >= 2
                and len(Xplot_compare) == len(Yplot_compare)
            )
            for i in range(len(words)):
                try:
                    m = _df_word_metric_mask(df["Word"], words[i])
                    df.loc[m, "Digitized Value (ray)"] = values[i]
                except Exception:
                    continue

            if has_compare:
                x_c = np.asarray(Xplot_compare, dtype=float)
                y_c = np.asarray(Yplot_compare, dtype=float)
                peaks_compare_m, troughs_compare_m, vals_compare = _digitized_peaks_metrics_timescaled(
                    x_c, y_c, hr, arbitrary_period
                )
                if vals_compare:
                    compare_values = vals_compare
                    if _is_ophthalmic_df(df):
                        compare_values = [
                            vals_compare[0],
                            vals_compare[1],
                            vals_compare[2],
                            vals_compare[3],
                            vals_compare[5],
                        ]
                    for i in range(len(words)):
                        try:
                            m = _df_word_metric_mask(df["Word"], words[i])
                            df.loc[m, "Digitized Value (morph)"] = compare_values[i]
                        except Exception:
                            continue

            has_grow = (
                Xplot_grow is not None
                and Yplot_grow is not None
                and len(Xplot_grow) >= 2
                and len(Xplot_grow) == len(Yplot_grow)
            )
            if has_grow:
                x_grow = np.asarray(Xplot_grow, dtype=float)
                y_grow = np.asarray(Yplot_grow, dtype=float)
                peaks_grow_m, troughs_grow_m, vals_grow = _digitized_peaks_metrics_timescaled(
                    x_grow, y_grow, hr, arbitrary_period
                )
                if vals_grow:
                    grow_values = vals_grow
                    if _is_ophthalmic_df(df):
                        grow_values = [
                            vals_grow[0],
                            vals_grow[1],
                            vals_grow[2],
                            vals_grow[3],
                            vals_grow[5],
                        ]
                    for i in range(len(words)):
                        try:
                            m = _df_word_metric_mask(df["Word"], words[i])
                            df.loc[m, "Digitized Value (grow)"] = grow_values[i]
                        except Exception:
                            continue

    except Exception:
        traceback.print_exc()
        arbitrary_period = 1.0

    # Plot digitization: usable beat markers only. Time-scaled x when HR valid;
    # arbitrary x when HR invalid so markers are still drawn.
    # -------------------------------------------------------------------------
    try:
        scale_factor, hr, _ = digitized_hr_scale_factor_for_raster(Xplot, Yplot, df)
        if np.isfinite(hr) and hr > 0.0:
            x_plot = x * scale_factor
            xlabel = "Time (s)"
        else:
            logger.warning("Invalid HR value for time scaling (%s); plotting on arbitrary x with beat markers.", hr)
            x_plot = x
            xlabel = "Arbitrary time scale"

        use_compare = (
            Xplot_compare is not None
            and Yplot_compare is not None
            and len(Xplot_compare) >= 2
            and len(Xplot_compare) == len(Yplot_compare)
        )
        x_plot_o = None
        y_o = None
        if use_compare:
            x_o = np.asarray(Xplot_compare, dtype=float)
            y_o = np.asarray(Yplot_compare, dtype=float)
            if np.isfinite(hr) and hr > 0.0:
                x_plot_o = x_o * scale_factor
            else:
                x_plot_o = x_o

        use_grow_plot = (
            Xplot_grow is not None
            and Yplot_grow is not None
            and len(Xplot_grow) >= 2
            and len(Xplot_grow) == len(Yplot_grow)
        )
        x_plot_g = None
        y_g = None
        if use_grow_plot:
            x_g_src = np.asarray(Xplot_grow, dtype=float)
            y_g = np.asarray(Yplot_grow, dtype=float)
            if np.isfinite(hr) and hr > 0.0:
                x_plot_g = x_g_src * scale_factor
            else:
                x_plot_g = x_g_src

        plt.close(2)
        fig2 = plt.figure(2)
        fig2.clf()
        if SHOW_BEAT_DEBUG_SUBPLOTS and use_compare and x_plot_o is not None:
            gs = fig2.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.12)
            ax_sig = fig2.add_subplot(gs[0, 0])
            ax_d3_ray = fig2.add_subplot(gs[1, 0], sharex=ax_sig)
            ax_d3_morph = fig2.add_subplot(gs[2, 0], sharex=ax_sig)
        elif SHOW_BEAT_DEBUG_SUBPLOTS:
            gs = fig2.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.12)
            ax_sig = fig2.add_subplot(gs[0, 0])
            ax_d3_ray = fig2.add_subplot(gs[1, 0], sharex=ax_sig)
            ax_d3_morph = None
        else:
            ax_sig = fig2.add_subplot(1, 1, 1)
            ax_d3_ray = None
            ax_d3_morph = None
        ray_color = "red"
        morph_color = MORPH_CURVE_COLOR
        if use_compare and x_plot_o is not None:
            ax_sig.plot(x_plot, y, "-", color=ray_color, linewidth=1.8, label="ray")
            ax_sig.plot(x_plot_o, y_o, "-", color=morph_color, linewidth=1.8, label="morph")
        else:
            ax_sig.plot(x_plot, y, "-", color=ray_color, linewidth=1.8, label="ray")
        if use_grow_plot and x_plot_g is not None and y_g is not None:
            ax_sig.plot(
                x_plot_g,
                y_g,
                "-",
                color=GROW_CURVE_COLOR,
                linewidth=1.6,
                label="grow",
            )

        # ---------------------------------------------------------------------
        # Visualise EXACT detection windows/anchors/feet by using the same peak
        # proposer and _detect_feet_indices_from_peaks debug records.
        # ---------------------------------------------------------------------
        def _anchors_and_debug(x_ser, y_ser, min_distance):
            y_arr = np.asarray(y_ser, dtype=float)
            try:
                amp = float(np.max(y_arr) - np.min(y_arr))
                prom = amp / 4.0 if np.isfinite(amp) and amp > 0 else None
                anchor_peaks, _ = find_peaks(
                    y_arr, distance=int(min_distance), prominence=prom
                )
            except Exception:
                anchor_peaks, _unused_tr = _beat_detection_pass(x_ser, y_ser, min_distance)
            if anchor_peaks is None:
                return np.array([], dtype=int), np.array([], dtype=int), []
            anchor_peaks = np.asarray(anchor_peaks, dtype=int)
            if anchor_peaks.size == 0:
                return anchor_peaks, np.array([], dtype=int), []
            feet_ser, debug_rows = _detect_feet_indices_from_peaks(
                x_ser, y_ser, anchor_peaks, return_debug=True
            )
            return anchor_peaks, np.asarray(feet_ser, dtype=int), list(debug_rows)

        anchor_peaks_primary, feet_plot_primary, debug_rows_primary = _anchors_and_debug(
            x_plot, y, max(1, len(x_plot) // 15)
        )
        anchor_peaks_compare = np.array([], dtype=int)
        feet_plot_compare = np.array([], dtype=int)
        debug_rows_compare = []
        if use_compare and x_plot_o is not None and y_o is not None and len(x_plot_o) == len(y_o):
            anchor_peaks_compare, feet_plot_compare, debug_rows_compare = _anchors_and_debug(
                x_plot_o, y_o, max(1, len(x_plot_o) // 15)
            )

        anchor_peaks_grow = np.array([], dtype=int)
        feet_plot_grow = np.array([], dtype=int)
        if use_grow_plot and x_plot_g is not None and y_g is not None and len(x_plot_g) == len(y_g):
            anchor_peaks_grow, feet_plot_grow, _debug_rows_grow = _anchors_and_debug(
                x_plot_g, y_g, max(1, len(x_plot_g) // 15)
            )

        # Shade windows + plot anchor peaks (on top of the waveform)
        primary_color = ray_color
        compare_color = morph_color

        if SHOW_BEAT_DEBUG_SUBPLOTS:
            for k, row in enumerate(debug_rows_primary):
                s = int(row["search_start"])
                e = int(row["search_end"])
                ax_sig.axvspan(
                    x_plot[s],
                    x_plot[e - 1],
                    color=primary_color,
                    alpha=0.06,
                    label="foot search (primary)" if k == 0 else None,
                )
            if anchor_peaks_primary.size > 0:
                ax_sig.plot(
                    x_plot[anchor_peaks_primary],
                    y[anchor_peaks_primary],
                    "x",
                    color=primary_color,
                    markersize=5,
                    zorder=8,
                    label="anchor peaks (primary)",
                )
            if use_compare and x_plot_o is not None and y_o is not None and anchor_peaks_compare.size > 0:
                for k, row in enumerate(debug_rows_compare):
                    s = int(row["search_start"])
                    e = int(row["search_end"])
                    ax_sig.axvspan(
                        x_plot_o[s],
                        x_plot_o[e - 1],
                        color=compare_color,
                        alpha=0.06,
                        label="foot search (compare)" if k == 0 else None,
                    )
                ax_sig.plot(
                    x_plot_o[anchor_peaks_compare],
                    y_o[anchor_peaks_compare],
                    "x",
                    color=compare_color,
                    markersize=5,
                    zorder=8,
                    label="anchor peaks (compare)",
                )
            if feet_plot_primary is not None and len(feet_plot_primary) > 0:
                ax_sig.plot(
                    x_plot[feet_plot_primary],
                    y[feet_plot_primary],
                    "o",
                    markerfacecolor="white",
                    markeredgecolor=primary_color,
                    markersize=5,
                    linewidth=0,
                    zorder=9,
                    label="feet (primary)",
                )
            if feet_plot_compare is not None and len(feet_plot_compare) > 0 and x_plot_o is not None and y_o is not None:
                ax_sig.plot(
                    x_plot_o[feet_plot_compare],
                    y_o[feet_plot_compare],
                    "o",
                    markerfacecolor="white",
                    markeredgecolor=compare_color,
                    markersize=5,
                    linewidth=0,
                    zorder=9,
                    label="feet (compare)",
                )

        # Visual PS fallback for plotting: if metrics PS are empty, use anchor peaks so
        # the top plot still shows systolic candidates for debugging.
        if (peaks_for_metrics is None) or (len(peaks_for_metrics) == 0):
            peaks_for_metrics = np.asarray(anchor_peaks_primary, dtype=int)
        # Visual ED fallback for plotting: if ED points are sparse/missing, use feet
        # (exclude last foot) as diastolic anchors so markers remain visible.
        if (troughs_for_metrics is None) or (len(troughs_for_metrics) == 0):
            if feet_plot_primary is not None and len(feet_plot_primary) >= 2:
                troughs_for_metrics = np.asarray(feet_plot_primary[:-1], dtype=int)
            elif feet_plot_primary is not None and len(feet_plot_primary) == 1:
                troughs_for_metrics = np.asarray(feet_plot_primary, dtype=int)
        if use_compare and x_plot_o is not None and (
            peaks_compare_m is None or len(peaks_compare_m) == 0
        ):
            peaks_compare_m = np.asarray(anchor_peaks_compare, dtype=int)
        if use_compare and x_plot_o is not None and (
            troughs_compare_m is None or len(troughs_compare_m) == 0
        ):
            if feet_plot_compare is not None and len(feet_plot_compare) >= 2:
                troughs_compare_m = np.asarray(feet_plot_compare[:-1], dtype=int)
            elif feet_plot_compare is not None and len(feet_plot_compare) == 1:
                troughs_compare_m = np.asarray(feet_plot_compare, dtype=int)
        if use_grow_plot and x_plot_g is not None and (
            peaks_grow_m is None or len(peaks_grow_m) == 0
        ):
            peaks_grow_m = np.asarray(anchor_peaks_grow, dtype=int)
        if use_grow_plot and x_plot_g is not None and (
            troughs_grow_m is None or len(troughs_grow_m) == 0
        ):
            if feet_plot_grow is not None and len(feet_plot_grow) >= 2:
                troughs_grow_m = np.asarray(feet_plot_grow[:-1], dtype=int)
            elif feet_plot_grow is not None and len(feet_plot_grow) == 1:
                troughs_grow_m = np.asarray(feet_plot_grow, dtype=int)

        # Beat delimiters on saved plot: faint dashed vertical lines at foot boundaries.
        # When morph (compare) is present, use morph feet and x so delimiters match the morph trace.
        _x_delim = x_plot
        _feet_delim = feet_plot_primary
        _ap_delim = anchor_peaks_primary
        if (
            use_compare
            and x_plot_o is not None
            and y_o is not None
            and feet_plot_compare is not None
            and len(feet_plot_compare) >= 2
        ):
            _fc = np.asarray(feet_plot_compare, dtype=int)
            _fc = _fc[(_fc >= 0) & (_fc < len(x_plot_o))]
            if _fc.size >= 2:
                _x_delim = x_plot_o
                _feet_delim = _fc
                _ap_delim = anchor_peaks_compare

        if _feet_delim is not None and len(_feet_delim) >= 2:
            fb = np.asarray(_feet_delim, dtype=int)
            fb = fb[(fb >= 0) & (fb < len(_x_delim))]
            for kk, fi in enumerate(fb):
                ax_sig.axvline(
                    _x_delim[int(fi)],
                    linestyle="--",
                    linewidth=0.8,
                    color="0.45",
                    alpha=0.30,
                    zorder=1,
                    label="beat boundary" if kk == 0 else None,
                )

        # Light red shading for incomplete edge regions (only if incomplete).
        # Start is incomplete when the first detected foot is not near the left edge.
        # End is incomplete when there is an anchor peak after the last detected foot.
        if _feet_delim is not None and len(_feet_delim) > 0:
            fp = np.asarray(_feet_delim, dtype=int)
            fp = fp[(fp >= 0) & (fp < len(_x_delim))]
            if fp.size > 0:
                # Incomplete first wave
                if int(fp[0]) > 0:
                    ax_sig.axvspan(
                        _x_delim[0],
                        _x_delim[int(fp[0])],
                        color="#ff6b6b",
                        alpha=0.08,
                        zorder=0,
                        label="incomplete region",
                    )
                # Incomplete last wave
                if _ap_delim is not None and len(_ap_delim) > 0:
                    ap = np.asarray(_ap_delim, dtype=int)
                    ap = ap[(ap >= 0) & (ap < len(_x_delim))]
                    if ap.size > 0 and int(ap[-1]) > int(fp[-1]):
                        ax_sig.axvspan(
                            _x_delim[int(fp[-1])],
                            _x_delim[-1],
                            color="#ff6b6b",
                            alpha=0.08,
                            zorder=0,
                            label=None,
                        )

        # Derivative diagnostic subplots: show the selector signal (dy) and d2 anchor.
        if SHOW_BEAT_DEBUG_SUBPLOTS:
            try:
                def _plot_dy_panel(
                    ax,
                    x_ser,
                    y_ser,
                    debug_rows,
                    line_color,
                    label_name,
                    ax_signal=None,
                    signal_marker_label=None,
                ):
                    if ax is None or x_ser is None or y_ser is None:
                        return
                    ax.axhline(0.0, color="0.75", linewidth=1)
                    d2_anchor_x = []
                    d2_anchor_y = []
                    foot_x = []
                    foot_y = []
                    for k, row in enumerate(debug_rows):
                        s = int(row.get("search_start", -1))
                        e = int(row.get("search_end", -1))
                        if s < 0 or e <= s or e > len(y_ser):
                            continue
                        x_r = np.asarray(x_ser[s:e], dtype=float)
                        dy_r = np.asarray(row.get("dy", []), dtype=float)
                        if dy_r.size != (e - s):
                            continue
                        ax.plot(
                            x_r, dy_r, "-", color=line_color, linewidth=1.1, alpha=0.9,
                            label=f"dy/dx ({label_name})" if k == 0 else None,
                        )
                        ax.axvspan(
                            x_r[0], x_r[-1], color=line_color, alpha=0.06,
                            label=f"foot search ({label_name})" if k == 0 else None,
                        )
                        ub_global = int(row.get("foot2_global", -1))
                        if ub_global >= s and ub_global < e:
                            ub_local = int(ub_global - s)
                            d2_anchor_x.append(float(x_ser[ub_global]))
                            d2_anchor_y.append(float(dy_r[ub_local]))
                        picked_global = int(row.get("picked_global", -1))
                        if picked_global >= s and picked_global < e:
                            picked_local = int(picked_global - s)
                            foot_x.append(float(x_ser[picked_global]))
                            foot_y.append(float(dy_r[picked_local]))
                        slope_thr = float(row.get("slope_thr", np.nan))
                        if np.isfinite(slope_thr):
                            ax.plot(
                                [x_r[0], x_r[-1]], [slope_thr, slope_thr], ":",
                                color=line_color, linewidth=0.8, alpha=0.5,
                                label=f"dy threshold ({label_name})" if k == 0 else None,
                            )
                    if len(foot_x) > 0:
                        ax.scatter(np.asarray(foot_x), np.asarray(foot_y), color=line_color, s=18, marker="D",
                                   label=f"selected foot (dy onset, {label_name})", zorder=5)
                    if len(d2_anchor_x) > 0:
                        ax.scatter(np.asarray(d2_anchor_x), np.asarray(d2_anchor_y), marker="s", facecolors="none",
                                   edgecolors=line_color, s=24, linewidths=1.0,
                                   label=f"d2 anchor ({label_name})", zorder=6)
                        if ax_signal is not None:
                            ub_idx = np.array([int(row.get("foot2_global", -1)) for row in debug_rows], dtype=int)
                            ub_idx = ub_idx[(ub_idx >= 0) & (ub_idx < len(y_ser))]
                            if ub_idx.size > 0:
                                ax_signal.scatter(
                                    x_ser[ub_idx], y_ser[ub_idx], marker="s", facecolors="none",
                                    edgecolors=line_color, s=26, linewidths=1.0,
                                    label=signal_marker_label, zorder=8,
                                )
                    ax.relim()
                    ax.autoscale_view()
                    ax.set_ylabel("dy/dx")
                    ax.set_title(f"Windowed dy/dx with d2 anchor ({label_name} series)")
                    ax.grid(True)
                    if len(ax.lines) > 0 or len(ax.collections) > 0:
                        ax.legend(loc="best", fontsize=8)

                if use_compare and x_plot_o is not None and y_o is not None:
                    _plot_dy_panel(ax_d3_ray, x_plot, y, debug_rows_primary, ray_color, "ray",
                                   ax_signal=ax_sig, signal_marker_label="d2 upper bound (ray)")
                    _plot_dy_panel(ax_d3_morph, x_plot_o, y_o, debug_rows_compare, morph_color, "morph",
                                   ax_signal=ax_sig, signal_marker_label="d2 upper bound (morph)")
                else:
                    _plot_dy_panel(ax_d3_ray, x_plot, y, debug_rows_primary, ray_color, "ray",
                                   ax_signal=ax_sig, signal_marker_label="d2 upper bound")
            except Exception:
                pass

        ax_sig.set_ylabel("Flowrate (cm/s)")
        if SHOW_BEAT_DEBUG_SUBPLOTS and ax_d3_morph is not None:
            ax_d3_morph.set_xlabel(xlabel)
        elif SHOW_BEAT_DEBUG_SUBPLOTS:
            ax_d3_ray.set_xlabel(xlabel)
        else:
            ax_sig.set_xlabel(xlabel)
        ax_sig.grid(True)

        # Marker diagnostics: missing markers usually mean beat detection failed
        # (too few peaks/feet) or compare series detection failed.
        if len(peaks_for_metrics) == 0 or len(troughs_for_metrics) == 0:
            logger.info(
                "plot_correction: primary beat markers missing (peaks=%d troughs=%d).",
                int(len(peaks_for_metrics)),
                int(len(troughs_for_metrics)),
            )
        if use_compare and x_plot_o is not None and (len(peaks_compare_m) == 0 or len(troughs_compare_m) == 0):
            logger.info(
                "plot_correction: compare beat markers missing (peaks=%d troughs=%d).",
                int(len(peaks_compare_m)),
                int(len(troughs_compare_m)),
            )
        if use_grow_plot and x_plot_g is not None and (
            len(peaks_grow_m) == 0 or len(troughs_grow_m) == 0
        ):
            logger.info(
                "plot_correction: grow beat markers missing (peaks=%d troughs=%d).",
                int(len(peaks_grow_m)),
                int(len(troughs_grow_m)),
            )

        if use_compare and x_plot_o is not None:
            if len(peaks_for_metrics) > 0:
                ax_sig.plot(
                    x_plot[peaks_for_metrics],
                    y[peaks_for_metrics],
                    "^",
                    color=ray_color,
                    alpha=0.35,
                    markersize=6,
                    zorder=4,
                    label="PS (ray)",
                )
            if len(troughs_for_metrics) > 0:
                ax_sig.plot(
                    x_plot[troughs_for_metrics],
                    y[troughs_for_metrics],
                    "v",
                    color=ray_color,
                    alpha=0.35,
                    markersize=6,
                    zorder=4,
                    label="ED (ray)",
                )
            if len(peaks_compare_m) > 0:
                ax_sig.plot(
                    x_plot_o[peaks_compare_m],
                    y_o[peaks_compare_m],
                    "^",
                    color=morph_color,
                    alpha=0.35,
                    markersize=6,
                    zorder=4,
                    label="PS (morph)",
                )
            if len(troughs_compare_m) > 0:
                ax_sig.plot(
                    x_plot_o[troughs_compare_m],
                    y_o[troughs_compare_m],
                    "v",
                    color=morph_color,
                    alpha=0.35,
                    markersize=6,
                    zorder=4,
                    label="ED (morph)",
                )
        else:
            if len(peaks_for_metrics) > 0:
                ax_sig.plot(
                    x_plot[peaks_for_metrics],
                    y[peaks_for_metrics],
                    "^",
                    color=ray_color,
                    alpha=0.35,
                    markersize=6,
                    zorder=4,
                    label="PS",
                )
            if len(troughs_for_metrics) > 0:
                ax_sig.plot(
                    x_plot[troughs_for_metrics],
                    y[troughs_for_metrics],
                    "v",
                    color=ray_color,
                    alpha=0.35,
                    markersize=6,
                    zorder=4,
                    label="ED",
                )
        if use_grow_plot and x_plot_g is not None and y_g is not None:
            if len(peaks_grow_m) > 0:
                ax_sig.plot(
                    x_plot_g[peaks_grow_m],
                    y_g[peaks_grow_m],
                    "^",
                    color=GROW_CURVE_COLOR,
                    alpha=0.35,
                    markersize=6,
                    zorder=4,
                    label="PS (grow)",
                )
            if len(troughs_grow_m) > 0:
                ax_sig.plot(
                    x_plot_g[troughs_grow_m],
                    y_g[troughs_grow_m],
                    "v",
                    color=GROW_CURVE_COLOR,
                    alpha=0.35,
                    markersize=6,
                    zorder=4,
                    label="ED (grow)",
                )
        # Build top legend AFTER all markers are added, with de-duplicated labels.
        try:
            handles, labels = ax_sig.get_legend_handles_labels()
            uniq = {}
            for h, l in zip(handles, labels):
                if l and l not in uniq:
                    uniq[l] = h
            if len(uniq) > 0:
                ax_sig.legend(
                    list(uniq.values()),
                    list(uniq.keys()),
                    loc="lower left",
                    ncol=2,
                    fontsize=7,
                    framealpha=0.92,
                )
        except Exception:
            pass
        x_lo = float(np.min(x_plot))
        x_hi = float(np.max(x_plot))
        if use_compare and x_plot_o is not None and y_o is not None and y_o.size:
            x_lo = min(x_lo, float(np.min(x_plot_o)))
            x_hi = max(x_hi, float(np.max(x_plot_o)))
        if use_grow_plot and x_plot_g is not None and y_g is not None and y_g.size:
            x_lo = min(x_lo, float(np.min(x_plot_g)))
            x_hi = max(x_hi, float(np.max(x_plot_g)))
        if np.isfinite(x_lo) and np.isfinite(x_hi) and x_hi > x_lo:
            ax_sig.set_xlim((x_lo, x_hi))
        y_hi = float(np.max(y))
        if use_compare and y_o is not None and y_o.size:
            y_hi = max(y_hi, float(np.max(y_o)))
        if use_grow_plot and y_g is not None and y_g.size:
            y_hi = max(y_hi, float(np.max(y_g)))
        ax_sig.set_ylim((0, y_hi + 10))
    except Exception:
        logger.warning("Could not plot digitization waveform; continuing.", exc_info=True)

    return df


def extract_dicom_label_text(cv2_img):
    """
    Extract vessel type (uterine/umbilical) and side (Rt/Lt) from yellow label text on the
    left side of a DICOM Doppler image. Uses HSV yellow mask and OCR on top 2/3, left half.

    Args:
        cv2_img (np.ndarray): BGR or grayscale image from the DICOM (e.g. from extract_doppler_from_dicom).

    Returns:
        dict: {"vessel": "Uterine"|"Umbilical"|None, "side": "Rt"|"Lt"|None, "raw": str}
    """
    result = {"vessel": None, "side": None, "raw": ""}
    if cv2_img is None or cv2_img.size == 0:
        return result

    img = np.asarray(cv2_img)
    h, w = img.shape[0], img.shape[1]
    # ROI: top 2/3 of height, left half of width (big yellow label lives there)
    roi_top = int(h * 2 / 3)
    roi_left = int(w / 2)

    if img.ndim == 2:
        gray = img
        yellow_roi = gray[0:roi_top, 0:roi_left]
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([1, 100, 100], dtype=np.uint8)
        upper_yellow = np.array([200, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        yellow_text = cv2.bitwise_and(gray, gray, mask=mask)
        yellow_roi = yellow_text[0:roi_top, 0:roi_left]

    try:
        text = pytesseract.image_to_string(yellow_roi, lang="eng", config="--oem 1 --psm 6")
    except Exception as e:
        logger.warning("DICOM label OCR failed: %s", e)
        return result

    text = (text or "").strip()
    result["raw"] = text
    # Use only letters and spaces for matching (exclude numbers and symbols)
    text_letters_only = re.sub(r"[^a-zA-Z\s]", " ", text)
    text_letters_only = re.sub(r"\s+", " ", text_letters_only).strip()
    text_lower = text_letters_only.lower()
    # Normalise common OCR substitutions (1 for i, 0 for o)
    text_normalised = text_lower.replace("1", "i").replace("0", "o")

    # Vessel: exact substring and whole-word only.
    # Do not use "art"/"artery" – they appear in both uterine and umbilical artery labels.
    if "uterine" in text_lower or "uterine" in text_normalised:
        result["vessel"] = "Uterine"
    elif (
        "umbilical" in text_lower
        or "umbilica" in text_lower
        or "umblical" in text_lower
        or "umbilical" in text_normalised
        or re.search(r"\bumb\b", text_lower)
        or "bilical" in text_lower
        or "bilical" in text_normalised
    ):
        result["vessel"] = "Umbilical"

    # Side: exact word boundaries only
    if re.search(r"\blt\b", text_lower) or " left" in text_lower or text_lower.startswith("left"):
        result["side"] = "Lt"
    elif re.search(r"\brt\b", text_lower) or " right" in text_lower or text_lower.startswith("right"):
        result["side"] = "Rt"

    # Log extracted text and what is matched for DICOM labels
    label_str = " ".join(filter(None, [result["side"], result["vessel"]])) or "(none)"
    logger.info(
        "DICOM label OCR extracted text %s; matched label %s",
        repr(text_letters_only or text),
        label_str,
    )

    return result


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
    ] + OPHTHALMIC_TARGET_WORDS

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


def _sorted_xy_points_pil_from_curve_coords(coords):
    """Curve coords as (row, col) -> PIL polyline [(x, y), ...] sorted along x."""
    if coords is None:
        return None
    arr = np.asarray(coords)
    if arr.size == 0 or arr.ndim != 2 or arr.shape[1] < 2:
        return None
    rows, cols = arr[:, 0], arr[:, 1]
    order = np.argsort(cols)
    pts = list(
        zip(cols[order].astype(int), rows[order].astype(int))
    )
    return pts if len(pts) >= 2 else None


def _draw_curve_polylines_rgba(
    img_rgba,
    top_curve_coords=None,
    ray_top_curve_coords=None,
    grow_top_curve_coords=None,
    morph_line_rgba=MORPH_LINE_RGBA,
    ray_line_rgba=(255, 0, 0, 255),
    grow_line_rgba=GROW_LINE_RGBA,
    line_width=2,
):
    """Overlay morph (blue), ray (red), and optional grow (violet) polylines on PIL RGBA."""
    draw = ImageDraw.Draw(img_rgba)
    pts_m = _sorted_xy_points_pil_from_curve_coords(top_curve_coords)
    if pts_m is not None:
        draw.line(pts_m, fill=morph_line_rgba, width=line_width)
    pts_r = _sorted_xy_points_pil_from_curve_coords(ray_top_curve_coords)
    if pts_r is not None:
        draw.line(pts_r, fill=ray_line_rgba, width=line_width)
    pts_g = _sorted_xy_points_pil_from_curve_coords(grow_top_curve_coords)
    if pts_g is not None:
        draw.line(pts_g, fill=grow_line_rgba, width=line_width)


def annotate(
        input_image_obj,
        refined_segmentation_mask,
        Left_dimensions,
        Right_dimensions,
        Waveform_dimensions,
        Left_axis,
        Right_axis,
        top_curve_coords=None,
        ray_top_curve_coords=None,
        grow_top_curve_coords=None,
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
        top_curve_coords (ndarray, optional): ``(row, col)`` morphological envelope curve.
        ray_top_curve_coords (ndarray, optional): ``(row, col)`` ray-traced curve.
        grow_top_curve_coords (ndarray, optional): ``(row, col)`` region-grow envelope curve.

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

            # Red fill for segmented waveform blob (disabled — use curve polylines only).
            # if refined_segmentation_mask[y, x] == 1:
            #     pixel_data[x, y] = (
            #         255,
            #         pixel_data[x, y][1],
            #         pixel_data[x, y][2],
            #         250,
            #     )  # Segmented waveform as Red
            if refined_segmentation_mask[y, x] == 2:
                pixel_data[x, y] = (1, 255, 1, 255)  # Set ROIs to blue
            elif Left_axis[y, x] == 255:
                pixel_data[x, y] = (255, 0, 255, 255)  # Set ROIs to blue
            elif Right_axis[y, x] == 255:
                pixel_data[x, y] = (255, 0, 255, 255)  # Set ROIs to blue
            else:
                pixel_data[x, y] = pixel_data[x, y]

    _draw_curve_polylines_rgba(
        img_RGB,
        top_curve_coords=top_curve_coords,
        ray_top_curve_coords=ray_top_curve_coords,
        grow_top_curve_coords=grow_top_curve_coords,
    )
    return img_RGB


def annotate_dicom(
    input_image_obj,
    refined_segmentation_mask,
    dicom_metadata,
    top_curve_coords=None,
    ray_top_curve_coords=None,
    grow_top_curve_coords=None,
):
    """
    DICOM-only annotation: overlay waveform segmentation and waveform ROI box
    on the Doppler image. No left/right axis ROIs or tick masks (DICOM has
    physical axes from metadata). Returns the annotated image for saving.

    Args:
        input_image_obj (PIL.Image.Image): Doppler image (RGB or L); converted to RGBA internally.
        refined_segmentation_mask (numpy.ndarray): Segmentation mask (1=waveform, same shape as image).
        dicom_metadata (dict): Must contain RegionLocationMinX0, RegionLocationMaxX1,
            RegionLocationMinY0, RegionLocationMaxY1 for the waveform box.
        top_curve_coords, ray_top_curve_coords, grow_top_curve_coords (ndarray, optional):
            ``(row, col)`` curves to draw.

    Returns:
        PIL.Image.Image: Annotated image (RGBA) with waveform in red, ROI outline in green.
    """
    # Ensure we have a copy and RGBA so we can write (r,g,b,a) without errors
    if input_image_obj.mode != "RGBA":
        img = input_image_obj.convert("RGBA")
    else:
        img = input_image_obj.copy()

    Xmin = int(dicom_metadata.get("RegionLocationMinX0", 0))
    Xmax = int(dicom_metadata.get("RegionLocationMaxX1", 0))
    Ymin = int(dicom_metadata.get("RegionLocationMinY0", 0))
    Ymax = int(dicom_metadata.get("RegionLocationMaxY1", 0))

    # Work on a copy of the mask so we don't mutate the caller's array
    mask = np.asarray(refined_segmentation_mask, dtype=np.uint8).copy()
    h, w = mask.shape

    # Waveform ROI outline only (no left/right boxes)
    r = [Xmin, Xmax, Xmax, Xmin, Xmin]
    c = [Ymax, Ymax, Ymin, Ymin, Ymax]
    rr, cc = polygon_perimeter(c, r, shape=mask.shape)
    mask[rr, cc] = 2

    pixel_data = img.load()
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            if y >= h or x >= w:
                continue
            # Red tint for waveform mask interior (disabled — curve polylines only).
            # if mask[y, x] == 1:
            #     pixel_data[x, y] = (255, pixel_data[x, y][1], pixel_data[x, y][2], 250)
            if mask[y, x] == 2:
                pixel_data[x, y] = (1, 255, 1, 255)
            # else: leave pixel unchanged

    _draw_curve_polylines_rgba(
        img,
        top_curve_coords=top_curve_coords,
        ray_top_curve_coords=ray_top_curve_coords,
        grow_top_curve_coords=grow_top_curve_coords,
    )
    return img


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

    # Restrict text search to the right third of the image by zeroing
    # everything left of 2/3 width. This is in addition to the existing
    # vertical masking below.
    width, height = COL.size
    right_start = int(width * (2.0 / 3.0))
    for y in range(height):
        for x in range(right_start):
            PIX[x, y] = (0, 0, 0)

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
        # and also calculates the bounding box for each group of words.
        #
        # Empty / whitespace-only OCR entries are removed here so that they do
        # not participate in line grouping or later matching.
        valid_indices = [
            i for i, t in enumerate(OCR_data["text"])
            if t is not None and t != "" and t != " "
        ]
        if not valid_indices:
            return [], []

        words = [OCR_data["text"][i] for i in valid_indices]
        lefts = [OCR_data["left"][i] for i in valid_indices]
        tops = [OCR_data["top"][i] for i in valid_indices]
        widths = [OCR_data["width"][i] for i in valid_indices]
        heights = [OCR_data["height"][i] for i in valid_indices]
        x_centers = [left + width / 2 for left, width in zip(lefts, widths)]  # Calculate x_center for sorting

        # Use the corresponding y_center entries for valid words only.
        filtered_y_center = [y_center[i] for i in valid_indices]

        # Convert y_center to a numpy array and reshape for DBSCAN
        data = np.array(filtered_y_center).reshape(-1, 1)

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
        "Lt MCA-PS",
        "Lt MCA-ED",
        "Lt MCA-S/D",
        "Lt MCA-PI",
        "Lt MCA-RI",
        "Lt MCA-MD",
        "Lt MCA-TAmax",
        "Lt MCA-HR",
        "Rt MCA-PS",
        "Rt MCA-ED",
        "Rt MCA-S/D",
        "Rt MCA-PI",
        "Rt MCA-RI",
        "Rt MCA-MD",
        "Rt MCA-TAmax",
        "Rt MCA-HR",
    ] + OPHTHALMIC_TARGET_WORDS

    # Split text into lines (grouped boxes + full-string fallback for long labels)
    lines = list(grouped_words)
    string_lines = [
        ln.strip()
        for ln in pytesseract.image_to_string(
            pixels,
            lang="eng",
            config="--oem 1 --psm 3 -c tessedit_char_blacklist=l,!_|=$",
        ).splitlines()
        if ln.strip()
    ]
    combined_lines = string_lines + [ln for ln in lines if ln not in string_lines]

    def refine_hr_from_local_roi(df_in, lines_in, bboxes_in, col_img):
        """Re-read HR from a local ROI around the first-pass HR line.

        Uses first-pass matched HR line index -> line bounding box -> expanded ROI.
        Then runs a lightweight deterministic OCR pass focused on HR text.
        """
        if df_in is None or df_in.empty or "Word" not in df_in.columns or "Line" not in df_in.columns:
            return df_in

        hr_rows = df_in.index[df_in["Word"].str.contains("HR", na=False)].tolist()
        if not hr_rows:
            return df_in

        row_idx = hr_rows[0]
        line_number = df_in.loc[row_idx, "Line"]
        if pd.isna(line_number):
            return df_in
        line_idx = int(line_number) - 1
        if line_idx < 0 or line_idx >= len(lines_in) or line_idx >= len(bboxes_in):
            return df_in

        box = bboxes_in[line_idx]
        x1, y1 = box["top_left"]
        x2, y2 = box["bottom_right"]
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)

        # Expand ROI around first-pass HR line and clamp to image bounds.
        pad_x = max(15, int(0.35 * width))
        pad_y = max(10, int(0.80 * height))
        img_w, img_h = col_img.size
        rx1 = max(0, x1 - pad_x)
        ry1 = max(0, y1 - pad_y)
        rx2 = min(img_w, x2 + pad_x)
        ry2 = min(img_h, y2 + pad_y)
        if rx2 <= rx1 or ry2 <= ry1:
            return df_in

        roi = col_img.crop((rx1, ry1, rx2, ry2))
        roi_np = np.array(roi)
        if roi_np.size == 0:
            return df_in

        # Lightweight deterministic enhancement.
        if roi_np.ndim == 3:
            gray = cv2.cvtColor(roi_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = roi_np
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        up = cv2.resize(thr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        # Restrict extraction to HR-like tokens/lines.
        txt = pytesseract.image_to_string(
            up, lang="eng", config="--oem 1 --psm 7 -c tessedit_char_whitelist=HRhr0123456789./- bpmBPM"
        )
        txt_norm = re.sub(r"\s+", " ", (txt or "").strip())
        if "HR" not in txt_norm.upper():
            txt2 = pytesseract.image_to_string(
                up, lang="eng", config="--oem 1 --psm 6 -c tessedit_char_whitelist=HRhr0123456789./- bpmBPM"
            )
            txt2_norm = re.sub(r"\s+", " ", (txt2 or "").strip())
            if "HR" in txt2_norm.upper():
                txt_norm = txt2_norm
            else:
                return df_in

        # Parse HR value from local ROI text.
        matches = re.findall(r"(\d{2,3}(?:\.\d+)?)", txt_norm)
        if not matches:
            return df_in
        refined_candidates = [float(m) for m in matches]
        refined_candidates = [v for v in refined_candidates if 20 <= v <= 240]
        if not refined_candidates:
            return df_in
        hr_refined = refined_candidates[0]

        current_val = df_in.loc[row_idx, "Value"] if "Value" in df_in.columns else None

        def _hr_implausible(v):
            return v is None or pd.isna(v) or (not np.isfinite(float(v))) or float(v) < 20 or float(v) > 240

        if _hr_implausible(current_val):
            df_in.loc[row_idx, "Value"] = round(hr_refined, 2)
            if "Unit" in df_in.columns and (pd.isna(df_in.loc[row_idx, "Unit"]) or df_in.loc[row_idx, "Unit"] in ("", 0)):
                df_in.loc[row_idx, "Unit"] = "bpm"
            logger.warning(
                "hr_refine: replaced implausible/missing first-pass HR with local ROI OCR value %s",
                round(hr_refined, 2),
            )
        else:
            current_val_f = float(current_val)
            if abs(current_val_f - hr_refined) > 3:
                logger.warning(
                    "hr_refine: first-pass HR (%s) disagrees with local ROI OCR (%s); keeping first-pass value",
                    round(current_val_f, 2),
                    round(hr_refined, 2),
                )
        return df_in
    # Initialize DataFrame
    df = pd.DataFrame(columns=["Line", "Word", "Value", "Unit"])

    most_likely_prefix = _detect_metric_family_from_lines(combined_lines, data)
    if most_likely_prefix is not None:
        target_words = [word for word in target_words if _target_belongs_to_family(word, most_likely_prefix)]
        word_order = list(target_words)
    else:
        word_order = list(target_words)
        logger.warning(
            "Metric OCR: could not detect vessel family from OCR lines; "
            "using full target word list."
        )

    match_lines = combined_lines
    matched_lines = set()

    if most_likely_prefix in ("Lt Ophthalmic", "Rt Ophthalmic"):
        df, matched_lines, target_words = _match_ophthalmic_lines(match_lines, target_words)
    else:
        # Step 1: Exact matching
        for i, line in enumerate(match_lines):
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

    if most_likely_prefix not in ("Lt Ophthalmic", "Rt Ophthalmic"):
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
        for i, line in enumerate(match_lines):
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

        # If no line matched any target word exactly (distance == 0), flag it – this
        # is a strong indicator that the prefix matching is off for this scan.
        if not matched_lines:
            logger.warning(
                "Metric OCR: no exact prefix matches between OCR lines and target words; "
                "metric labels may be misaligned."
            )

        def find_closest_target(line, target_words):
            min_distance = float('inf')
            closest_word = None

            for word in target_words:
                # Compare only the prefix of the line up to the target word length,
                # so trailing numbers/units (e.g. " — 27.35cm/s") do not affect
                # the distance. Case and characters are preserved.
                candidate = line[: len(word)]
                distance = Levenshtein.distance(candidate, word)
                if distance < min_distance:
                    min_distance = distance
                    closest_word = word

            return closest_word, min_distance

        # Set a threshold for acceptable similarity
        threshold = 7

        # Step 3: Closest target word matching for unmatched lines
        for i, line in enumerate(match_lines):
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
        ] + OPHTHALMIC_TARGET_WORDS_EXTENDED

        if target_words:

            indices = [i for i, entry in enumerate(target_words_extended) if any(sub in entry for sub in target_words)]
            remaining_target_extended = [target_words_extended[i] for i in indices]
            remaining_target_words = []
            for entry in remaining_target_extended:
                matched_tw = next((tw for tw in target_words if tw in entry), None)
                if matched_tw is None:
                    matched_tw = target_words[min(len(remaining_target_words), len(target_words) - 1)]
                remaining_target_words.append(matched_tw)

            def _metric_suffix_for_fuzzy(word):
                if "-" in word:
                    return word.split("-")[-1]
                m = re.search(r"A\.\s*(\S+)", word)
                if m:
                    return m.group(1)
                return word.split()[-1]

            suffixes = [_metric_suffix_for_fuzzy(word) for word in remaining_target_extended]
            bias_dict = {suffix: -2 for suffix in suffixes}

            # Create a distance matrix
            num_lines = len(match_lines)
            num_target_words = len(remaining_target_extended)
            distance_matrix = np.zeros((num_lines, num_target_words))

            # Calculate biased distances
            for i, line in enumerate(match_lines):
                if i not in matched_lines:
                    for j, word in enumerate(remaining_target_extended):
                        # Remove digits from the line
                        line_no_digits = re.sub(r'\d+', '', line)

                        # Calculate basic Levenshtein distance
                        basic_distance = Levenshtein.distance(line_no_digits, word)

                        # Apply bias if a specific suffix is expected in the line
                        expected_suffix = suffixes[j]
                        if expected_suffix in line:
                            basic_distance += bias_dict[expected_suffix]

                        # Set the biased distance in the matrix
                        distance_matrix[i, j] = basic_distance

            matches = {}
            for j, word in enumerate(remaining_target_extended):
                # Find the line with the smallest non-zero distance for the current target word
                line_indices_with_non_zero_distances = np.where(distance_matrix[:, j] > 0)[0]
                if len(line_indices_with_non_zero_distances) > 0:
                    i = line_indices_with_non_zero_distances[np.argmin(distance_matrix[line_indices_with_non_zero_distances, j])]
                    line = match_lines[i]
                    # Add to matches
                    matches[line] = word

                    # Extract value and unit from the line and add to the DataFrame
                    match = re.search(r"(\-?\d+(\s*\d+)*\.\s*\d+|\-?\d+(\s*\d+)*)\s*([^\d\s]+)?$", line)
                    if match:
                        value = float(match.group(1).replace(' ', ''))
                        unit = match.group(4) if match.group(4) else ""
                        df.loc[len(df)] = {"Line": i + 1, "Word": remaining_target_words[j], "Value": value, "Unit": unit}
                    else:
                        df.loc[len(df)] = {"Line": i + 1, "Word": remaining_target_words[j], "Value": 0, "Unit": 0}
                    matched_lines.add(i)

                    # Remove the matched line from further consideration
                    distance_matrix[i, :] = np.inf
    elif most_likely_prefix in ("Lt Ophthalmic", "Rt Ophthalmic") and not matched_lines:
        logger.warning(
            "Metric OCR: ophthalmic family detected but no metric lines matched."
        )

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
                logger.info("Metric DV: swapped DV-S/a and DV-S values")

            if df.loc[df['Word'] == 'DV-a/S', 'Unit'].values[0] != '' and df.loc[df['Word'] == 'DV-a', 'Unit'].values[0] == '':
                # Storing temporary values for swapping
                temp = df.loc[df['Word'] == 'DV-a/S', 'Value'].values[0]
                temp_unit = df.loc[df['Word'] == 'DV-a/S', 'Unit'].values[0]
                df.loc[df['Word'] == 'DV-a/S', 'Value'] = df.loc[df['Word'] == 'DV-a', 'Value'].values[0]
                df.loc[df['Word'] == 'DV-a/S', 'Unit'] = df.loc[df['Word'] == 'DV-a', 'Unit'].values[0]
                df.loc[df['Word'] == 'DV-a', 'Value'] = temp
                df.loc[df['Word'] == 'DV-a', 'Unit'] = temp_unit

            df = metric_check_dv(df)  # handle the ductus venousus differently
        elif most_likely_prefix in ("Lt Ophthalmic", "Rt Ophthalmic"):
            df = metric_check_ophthalmic(df)
        elif most_likely_prefix is not None:
            df = metric_check(df)  # for left, right, and umbilical
    except Exception:
        logger.exception("Metric check failed for uterine/umbilical metrics")

    try:
        df = refine_hr_from_local_roi(df, lines, bounding_boxes, COL)
    except Exception:
        logger.exception("hr_refine: local HR refinement failed")

    # Enforce positive heart-rate values: OCR occasionally hallucinates a leading '-'
    if not df.empty and 'Value' in df.columns and 'Word' in df.columns:
        hr_mask = df['Word'].str.contains('HR', na=False) & df['Value'].notna()
        df.loc[hr_mask, 'Value'] = df.loc[hr_mask, 'Value'].abs()

    return Fail, df


def metric_check_ophthalmic(df):
    """Validate and lightly correct ophthalmic artery OCR metrics (PS, ED, PI, RI, PS/ED)."""
    if "Raw Value" not in df.columns:
        df["Raw Value"] = df["Value"].copy()

    words = df["Word"]

    def get_val(metric):
        mask = _df_word_metric_mask(words, metric)
        if mask.any():
            return float(df.loc[mask, "Value"].values[0])
        return None

    def set_val(metric, val):
        mask = _df_word_metric_mask(words, metric)
        if mask.any():
            df.loc[mask, "Value"] = val

    pi = get_val("PI")
    if pi is not None:
        set_val("PI", check_pi_value(pi))

    ps = get_val("PS")
    ed = get_val("ED")
    if ps is None and ed is None:
        return df

    ps = ps if ps is not None else 0.0
    ed = ed if ed is not None else 0.0

    if 250 <= ps <= 1000:
        ps = ps / 10
    if 1000 <= ps <= 10000:
        ps = ps / 100
    if 200 <= ed <= 1000:
        ed = ed / 10
    if 1000 <= ed <= 10000:
        ed = ed / 100

    set_val("PS", ps)
    set_val("ED", ed)

    if ed == 0:
        ed = float(np.finfo(float).eps)

    ratio_calc = ps / ed
    ri_calc = resistive_index_from_ps_ed(ps, ed)

    ratio_extracted = get_val("PS/ED")
    if ratio_extracted is None:
        ratio_extracted = get_val("S/D")
    ri_extracted = get_val("RI")

    if ratio_extracted is None or abs(ratio_calc - ratio_extracted) > 0.5:
        if _df_word_metric_mask(words, "PS/ED").any():
            set_val("PS/ED", round(ratio_calc, 2))
        elif _df_word_metric_mask(words, "ED/PS").any() and ratio_calc != 0:
            set_val("ED/PS", round(1.0 / ratio_calc, 2))

    if ri_extracted is None or abs(ri_calc - ri_extracted) > 0.1:
        set_val("RI", round(ri_calc, 2))

    return df


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

    # Preserve raw, pre-correction metric values in a separate column so that
    # the main "Value" field can be corrected without losing the originals.
    if "Raw Value" not in df.columns:
        df["Raw Value"] = df["Value"]

    def identify_prefix(lines):
        # Keep this list aligned with the OCR metric labels used in text extraction.
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
            "Lt MCA-PS",
            "Lt MCA-ED",
            "Lt MCA-S/D",
            "Lt MCA-PI",
            "Lt MCA-RI",
            "Lt MCA-MD",
            "Lt MCA-TAmax",
            "Lt MCA-HR",
            "Rt MCA-PS",
            "Rt MCA-ED",
            "Rt MCA-S/D",
            "Rt MCA-PI",
            "Rt MCA-RI",
            "Rt MCA-MD",
            "Rt MCA-TAmax",
            "Rt MCA-HR",
        ] + OPHTHALMIC_TARGET_WORDS
        valid_prefixes = ["Lt Ut", "Rt Ut", "Umb", "Lt MCA", "Rt MCA", "Lt Ophthalmic", "Rt Ophthalmic"]
        prf = None
        for prefix in valid_prefixes:
            if prefix in ("Lt Ophthalmic", "Rt Ophthalmic"):
                if lines["Word"].astype(str).str.contains("Ophthalmic|Opthalmic", regex=True, na=False).any():
                    if prefix == "Lt Ophthalmic" and (
                        lines["Word"].astype(str).str.contains("Lt Ophthalmic|Lt Opthalmic", regex=True, na=False).any()
                    ):
                        prf = prefix
                        break
                    if prefix == "Rt Ophthalmic" and (
                        lines["Word"].astype(str).str.contains("Rt Ophthalmic|Rt Opthalmic", regex=True, na=False).any()
                    ):
                        prf = prefix
                        break
            elif lines["Word"].str.contains(prefix, regex=False).any():
                prf = prefix
                break

        if prf is None:
            logger.warning("metric_check: metric prefix could not be detected; keeping extracted rows as-is")
            return None, []

        logger.info("metric_check: metric prefix detected %s", prf)
        filtered_target_words = [word for word in target_words if _target_belongs_to_family(word, prf)]
        return prf, filtered_target_words

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

    def normalize_tamax_sign(value_in, df_in):  # Flip TAmax sign only when TAmax is negative but MD, PS, and ED are all positive.

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
    df.loc[df['Word'].str.contains('TAmax'), 'Value'] = normalize_tamax_sign(TAmax, df)

    # Peak systolic
    PS = df.loc[df['Word'].str.contains('PS'), 'Value'].values[0] if df['Word'].str.contains('PS').any() else 0
    # End diastolic
    ED = df.loc[df['Word'].str.contains('ED'), 'Value'].values[0] if df['Word'].str.contains('ED').any() else 0

    def normalize_ps_ed_values(PS, ED, df):  # Decimal can be misread, so common sense check.

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

    PS, ED = normalize_ps_ed_values(PS, ED, df)  # sense check for pressures
    df.loc[df['Word'].str.contains('PS'), 'Value'] = PS
    df.loc[df['Word'].str.contains('ED'), 'Value'] = ED

    # Find S/D
    SoverD_calc = PS / ED
    # Find RI
    RI_calc = (PS - ED) / PS
    # Find TAmax (PS/ED-only surrogate; no waveform here — see ``tamax_from_ps_ed_approximation``)
    TAmax_calc = tamax_from_ps_ed_approximation(PS, ED)

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

        # Tolerance level for metrics directly derived from PS and ED
        tolerance1 = 0.5
        conditions_met = []
        for parameter, extracted_name in [('SoverD', 'SoverD_extracted'), ('RI', 'RI_extracted'), ('TAmax', 'TAmax_extracted')]:
            extracted_value = c_df['Extracted'][extracted_name]

            if extracted_value is not None:
                if parameter == 'TAmax':
                    ps_val = c_df['Extracted']['PS_extracted']
                    ed_val = c_df['Extracted']['ED_extracted']
                    if ps_val is not None and ed_val is not None and abs(ed_val) < abs(extracted_value) < abs(ps_val):
                        conditions_met.append('TAmax_bounds')
                    continue
                for row_name, calc_value in c_df.iloc[:, col].items():
                    if str(row_name).startswith(extracted_name[:-9]):  # If row name starts with the parameter name
                        if abs(calc_value - extracted_value) < tolerance1:
                            conditions_met.append(row_name)
                            break  # Exit the inner loop once a match is found

        return conditions_met

    conditions_met = Metric_comparison(comparison_dataframe, 1)

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
            TAmax_from_ED_from_SoverD = (
                tamax_from_ps_ed_approximation(PS, ED_from_SoverD) if ED_from_SoverD else None
            )
            TAmax_from_ED_from_RI = (
                tamax_from_ps_ed_approximation(PS, ED_from_RI) if ED_from_RI else None
            )

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
                TAmax_from_PS_from_SoverD = (
                    tamax_from_ps_ed_approximation(PS_from_SoverD, ED) if PS_from_SoverD else None
                )
                TAmax_from_PS_from_RI = (
                    tamax_from_ps_ed_approximation(PS_from_RI, ED) if PS_from_RI else None
                )

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
                    logger.info("Metric OCR: recalculated PS from RI as %s", new_value)
                    df.loc[df['Word'].str.contains('PS'), 'Value'] = round(new_value, 2)
                    PS = df.loc[df['Word'].str.contains('PS'), 'Value'].values[0]
                    ED = df.loc[df['Word'].str.contains('ED'), 'Value'].values[0]
                    # Find S/D
                    df.loc[df['Word'].str.contains('S/D'), 'Value'] = round(PS / ED, 2)
                    # Find RI
                    df.loc[df['Word'].str.contains('RI'), 'Value'] = round((PS - ED) / PS, 2)
                    # Find TAmax
                    df.loc[df['Word'].str.contains('TAmax'), 'Value'] = round(
                        tamax_from_ps_ed_approximation(PS, ED), 2
                    )

            elif len(conditions_met) > 0:

                row_name = conditions_met[0]

                parts = row_name.split('_from_')
                desired_row_name = parts[1] + '_from_' + parts[2]
                new_value = comparison_dataframe.loc[desired_row_name, 'Second_calc']
                # We have calculated the new PS!
                logger.info("Metric OCR: recalculated ED from RI as %s", new_value)
                df.loc[df['Word'].str.contains('ED'), 'Value'] = round(new_value, 2)
                PS = df.loc[df['Word'].str.contains('PS'), 'Value'].values[0]
                ED = df.loc[df['Word'].str.contains('ED'), 'Value'].values[0]
                # Find S/D
                df.loc[df['Word'].str.contains('S/D'), 'Value'] = round(PS / ED, 2)
                # Find RI
                df.loc[df['Word'].str.contains('RI'), 'Value'] = round((PS - ED) / PS, 2)
                # Find TAmax
                df.loc[df['Word'].str.contains('TAmax'), 'Value'] = round(
                    tamax_from_ps_ed_approximation(PS, ED), 2
                )

        except ZeroDivisionError:
            logger.error("Metric OCR: division by zero encountered while checking uterine/umbilical metrics")
    elif len(conditions_met) < 3:
        # At least 1 of the metrics is consistent, therefore PS and ED can be assumed to be correct,
        # Calculate the inconsistent metrics from the PS and ED calculations
        logger.warning("Metric OCR: at least one text extraction error, correcting uterine/umbilical metrics")

        if 'SoverD' not in conditions_met:
            # Find S/D
            df.loc[df['Word'].str.contains('S/D'), 'Value'] = round(PS / ED, 2)
        if 'RI' not in conditions_met:
            # Find RI
            df.loc[df['Word'].str.contains('RI'), 'Value'] = round((PS - ED) / PS, 2)
        if 'TAmax' not in conditions_met:
            # Find TAmax
            df.loc[df['Word'].str.contains('TAmax'), 'Value'] = round(
                tamax_from_ps_ed_approximation(PS, ED), 2
            )
    else:
        logger.info("Metric OCR: all uterine/umbilical metrics are self-consistent")

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

    # Preserve raw, pre-correction DV metric values in a separate column so
    # that the main "Value" field can be corrected without losing originals.
    if "Raw Value" not in df.columns:
        df["Raw Value"] = df["Value"]

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

    def normalize_tamax_sign(value, df):  # Flip DV-TAmax sign only when TAmax is negative but DV-S/a, DV-S, and DV-D are all positive.

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
    df.loc[df['Word'] == 'DV-TAmax', 'Value'] = normalize_tamax_sign(TAmax, df)

    # Peak systolic
    PS = df.loc[df['Word'] == 'DV-S', 'Value'].values[0]
    # End diastolic
    ED = df.loc[df['Word'] == 'DV-D', 'Value'].values[0]

    def normalize_ps_ed_values(PS, ED, df):  # Decimal can be misread, so common sense check.

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

    PS, ED = normalize_ps_ed_values(PS, ED, df)  # sense check for pressures

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
        logger.info("Metric OCR: metric comparison completed successfully")
    except Exception:
        conditions_met = None
        logger.exception("Metric OCR: metric comparison failed")

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
                    logger.info("Metric OCR DV: recalculated PS from PI as %s", new_value)
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
                logger.info("Metric OCR DV: recalculated a from PI as %s", new_value)
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
            logger.error("Metric OCR DV: division by zero encountered while checking DV metrics")
    elif len(conditions_met) < 3:
        # At least 1 of the metrics is consistent, therefore PS and a can be assumed to be correct,
        # Calculate the inconsistent metrics from the PS and a calculations
        logger.warning("Metric OCR DV: at least one text extraction error, correcting DV metrics")

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
        logger.info("Metric OCR DV: all DV metrics are self-consistent")

    return df
