# /usr/bin/env python3

"""Segments the ultrasound images"""

# Python imports
import os
import logging
from PIL import Image
import pickle

# Module imports
import matplotlib.pyplot as plt
import cv2
import numpy as np
import traceback
import toml

# Import segmentation module
from usseg import general_functions
from usseg.setup_environment import setup_tesseract

logger = logging.getLogger(__file__)


def segment(filenames=None, output_dir=None, pickle_path=None):
    """Segments the pre-selected ultrasound images

    Args:
        filenames (str or list, optional) : If string, must be either a single
            file name path or a path to a pickle object containing the list of
            files. Pickle objects are expected to have the extension ".pkl"
            or ".pickle".
            If a list, must be a list of filenames to ultrasound images to
            segment.
            If None, will load a test image.

        output_dir (str, optional) : Path to the output directory to store annoated
            images. If None, will load from config file.
            Defaults to None.
        pickle_path (str or bool) : If pickle_path is False, will not store the
            list of likely us images as a pickle file. If None,
            will load the pickle path from "config.toml".
            Else if a string, will dump the pickled list to the specified path.
            Defaults to None.
    Returns:
        (tuple): tuple containing:
            - **filenames** (list): A list of the paths to the images that were segmented.
            - **Digitized_scans** (list): A list of the paths to the digitized scans.
            - **Annotated_scans** (list): A list of the paths to the annotated scans.
            - **Text_data** (list): A list of the text data extracted from the scans, as strings.
    """

    if filenames is None:
        filenames = ["Lt_test_image.png"]

    elif isinstance(filenames, list):
        pass

    elif isinstance(filenames, dict) or filenames.endswith(".pkl") or filenames.endswith(".pickle"):
        if isinstance(filenames, str):
            with open(filenames, "rb") as f:
                text_file = pickle.load(f)
        else:
            text_file = filenames

        # Get a list of all the keys in the dictionary
        subkeys = list(text_file.keys())

        filenames = []
        # Iterate through the sublist of keys
        for key in subkeys:
            # Access the value corresponding to the key
            filenames = filenames + text_file[key]
            #
    elif isinstance(filenames, str):
        filenames = [filenames]
    else:
        logging.warning(
            f"Unrecognised filenames type {type(filenames)}"
            "Excepted either a string or a list"
        )

    if output_dir is None:
        output_dir = toml.load("config.toml")["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    # excel_file = output_dir + "sample3_processed_data"
    Text_data = []  # text data extracted from image
    Annotated_scans = []
    Digitized_scans = []

    for input_image_filename in filenames:  # Iterate through all file names and populate excel file
        # input_image_filename = "E:/us-data-anon/0000/IHE_PDI/00003511/AA3A43F2/AAD8766D/0000371E\\EEEAE224.JPG"
        image_name = os.path.basename(input_image_filename)
        print(input_image_filename)

        # Classify file type for downstream handling (image vs DICOM)
        ext = os.path.splitext(input_image_filename)[1].lower()
        us_dicom = ext in (".dcm", ".dicom")
        # is_image: jpeg, png, or other common image formats
        us_image = ext in (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp")

        if us_dicom:
            # DICOM files skip text extraction, but extract metadata
            logger.info(f"Processing DICOM file: {input_image_filename}")
            dicom_metadata = general_functions.extract_dicom_metadata(input_image_filename)
            PIL_image, cv2_image = general_functions.extract_doppler_from_dicom(input_image_filename)
            general_functions.debug_plot_doppler_overlay(PIL_image, dicom_metadata, ref_is_relative=True)

            Text_data.append(None)
            Fail = 0
        elif us_image:
            try:  # Try text extraction
                colRGBA = Image.open(input_image_filename)  # These images are in RGBA form
                # colRGBA = General_functions.upscale_to_fixed_longest_edge(colRGBA)  # upscale to longest edge
                PIL_col = colRGBA.convert("RGB")  # We need RGB, so convert here. with PIL
                cv2_img = cv2.imread(input_image_filename)  # with cv2.
                # pix = (
                #     col.load()
                # )  # Loads a pixel access object, where pixel values can be edited

                # from General_functions import Colour_extract, Text_from_greyscale
                COL = general_functions.colour_extract_vectorized(PIL_col, [255, 255, 100], 95, 95)
                logger.info("Done Colour extract")

                Fail, df = general_functions.text_from_greyscale(cv2_img, COL)
            except Exception:  # flat fail on 1
                traceback.print_exc()  # prints the error message and traceback
                logger.error("Failed Text extraction")
                Text_data.append(None)
                Fail = 0
                pass
        else:
            # Unknown file type
            logger.warning(f"Unknown file type for {input_image_filename}, skipping text extraction")
            Text_data.append(None)
            Fail = 0

        if us_image:
            try:  # Try initial segmentation
                segmentation_mask, Xmin, Xmax, Ymin, Ymax = general_functions.initial_segmentation(
                    input_image_obj=PIL_col
                )
            except Exception:  # flat fail on 1
                logger.error("Failed Initial segmentation")
                Fail = Fail + 1
                pass

            try:  # define end ROIs
                Left_dimensions, Right_dimensions = general_functions.define_end_rois(
                    segmentation_mask, Xmin, Xmax, Ymin, Ymax
                )
            except Exception:
                logger.error("Failed Defining ROI")
                Fail = Fail + 1
                pass
        elif us_dicom:
            # DICOM: extract bounding box coordinates from metadata
            segmentation_mask = None
            Xmin = dicom_metadata.get("RegionLocationMinX0")
            Xmax = dicom_metadata.get("RegionLocationMaxX1")
            Ymin = dicom_metadata.get("RegionLocationMinY0")
            Ymax = dicom_metadata.get("RegionLocationMaxY1")
            Left_dimensions = Right_dimensions = None
        else:
            # Unknown file type: skip segmentation and ROI definition
            segmentation_mask = None
            Xmin = Xmax = Ymin = Ymax = None
            Left_dimensions = Right_dimensions = None

        try:
            Waveform_dimensions = [Xmin, Xmax, Ymin, Ymax]
        except Exception:
            logger.error("Failed Waveform dimensions")
            Fail = Fail + 1
            pass

        # Initialise axis tick/value and mask containers so that downstream
        # logic can safely handle cases where one side fails to be detected.
        Lnumber = None
        Lpositions = None
        Rnumber = None
        Rpositions = None
        ROIL = None
        ROIR = None
        y_zero = None

        try:  # Search for ticks and labels - Left axis
            (
                Cs,
                ROIAX,
                CenPoints,
                onY,
                BCs,
                TYLshift,
                thresholded_image,
                Side,
                Left_dimensions,
                Right_dimensions,
                ROI2,
                ROI3,
            ) = general_functions.search_for_ticks(
                cv2_img, "Left", Left_dimensions, Right_dimensions
            )
            ROIAX, Lnumber, Lpositions, ROIL = general_functions.search_for_labels(
                Cs,
                ROIAX,
                CenPoints,
                onY,
                BCs,
                TYLshift,
                Side,
                Left_dimensions,
                Right_dimensions,
                cv2_img,
                ROI2,
                ROI3,
            )
            # Validate and lightly clean left-axis ticks/labels
            Lnumber, Lpositions = general_functions.validate_axis_ticks(
                Lnumber, Lpositions, side="Left"
            )
        except Exception:
            traceback.print_exc()  # prints the error message and traceback
            logger.error("Failed Left Axes search")

            Fail = Fail + 1
            pass

        try:  # Search for ticks and labels - Right axis
            (
                Cs,
                ROIAX,
                CenPoints,
                onY,
                BCs,
                TYLshift,
                thresholded_image,
                Side,
                Left_dimensions,
                Right_dimensions,
                ROI2,
                ROI3,
            ) = general_functions.search_for_ticks(
                cv2_img, "Right", Left_dimensions, Right_dimensions
            )
            ROIAX, Rnumber, Rpositions, ROIR = general_functions.search_for_labels(
                Cs,
                ROIAX,
                CenPoints,
                onY,
                BCs,
                TYLshift,
                Side,
                Left_dimensions,
                Right_dimensions,
                cv2_img,
                ROI2,
                ROI3,
            )
            # Validate and lightly clean right-axis ticks/labels
            Rnumber, Rpositions = general_functions.validate_axis_ticks(
                Rnumber, Rpositions, side="Right"
            )
        except Exception:
            traceback.print_exc()  # prints the error message and traceback
            logger.error("Failed Right Axes search")

            Fail = Fail + 1
            pass

        # Cross-check consistency between left and right axes where both exist
        try:
            axis_agree = None

            # Only attempt a strict left/right consistency check if both sides
            # actually produced ticks and positions.
            if (
                Lnumber is not None
                and Lpositions is not None
                and Rnumber is not None
                and Rpositions is not None
            ):
                axis_agree = general_functions.validate_axis_pair(
                    Lnumber, Lpositions, Rnumber, Rpositions
                )

            # Estimate a global y=0 line (in image coordinates), using whatever
            # sides are available. The helper handles None gracefully.
            y_zero = general_functions.estimate_zero_line_y(
                left_numbers=Lnumber,
                left_positions=Lpositions,
                right_numbers=Rnumber,
                right_positions=Rpositions,
            )

            if y_zero is not None:
                logger.info(f"Estimated y=0 line at y={y_zero:.2f} for {input_image_filename}")
        except Exception:
            # These checks are purely diagnostic; never break the pipeline
            traceback.print_exc()


        try:
            try:  # Refine segmentation
                (
                    refined_segmentation_mask, top_curve_mask, top_curve_coords
                ) = general_functions.segment_refinement(
                    cv2_img, Xmin, Xmax, Ymin, Ymax, y_zero=y_zero
                )
            except Exception:
                traceback.print_exc()  # prints the error message and traceback
                logger.error("Failed Segment refinement")
                Fail = Fail + 1
                pass

            Xplot, Yplot, Ynought = general_functions.plot_digitized_data_single_axis(
                Rnumber, Rpositions, Lnumber, Lpositions, top_curve_coords,
            )

            # Ensure axis masks are always defined. If either side failed axis
            # detection, fall back to an empty mask so that annotation still
            # runs without raising NameError / index errors.
            if ROIL is None:
                ROIL = np.zeros_like(refined_segmentation_mask, dtype=np.uint8)
            if ROIR is None:
                ROIR = np.zeros_like(refined_segmentation_mask, dtype=np.uint8)

            col = general_functions.annotate(
                input_image_obj=colRGBA,
                refined_segmentation_mask=refined_segmentation_mask,
                Left_dimensions=Left_dimensions,
                Right_dimensions=Right_dimensions,
                Waveform_dimensions=Waveform_dimensions,
                Left_axis=ROIL,
                Right_axis=ROIR,
            )
            Annotated_path = output_dir + image_name.partition(".")[0] + "_Annotated.png"
            fig1, ax1 = plt.subplots(1)
            ax1.imshow(col)
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.tick_params(axis="both", which="both", length=0)
            fig1.savefig(Annotated_path, dpi=900, bbox_inches="tight", pad_inches=0)
            Annotated_scans.append(Annotated_path)

            try:
                df = general_functions.plot_correction(Xplot, Yplot, df)
                Text_data.append(df)
            except Exception:
                traceback.print_exc()
                logger.error("Failed correction")
                continue
            Digitized_path = output_dir + image_name.partition(".")[0] + "_Digitized.png"
            plt.figure(2)
            plt.savefig(Digitized_path, dpi=900, bbox_inches="tight", pad_inches=0)
            Digitized_scans.append(Digitized_path)

        except Exception:
            logger.error("Failed Digitization")
            Annotated_scans.append(None)
            traceback.print_exc()
            try:
                Text_data.append(df)
            except Exception:
                traceback.print_exc()
                Text_data.append(None)
            Digitized_scans.append(None)
            Fail = Fail + 1
            pass

        to_del = [
            "df",
            "image_name",
            "Xmax",
            "Xmin",
            "Ymax",
            "Ymin",
            "Rnumber",
            "Rpositions",
            "Lnumber",
            "Lpositions",
            "Left_dimensions",
            "Right_dimensions",
            "segmentation_mask",
        ]
        for i in to_del:
            try:
                exec("del %s" % i)
            except Exception:
                pass

        plt.close("all")
        i = 1

    print(Digitized_scans)
    print(Annotated_scans)
    print(Text_data)
    if pickle_path is not False:
        if pickle_path is None:
            pickle_path = toml.load("config.toml")["pickle"]["segmented_data"]
        with open(pickle_path, "wb") as f:
            pickle.dump([filenames, Digitized_scans, Annotated_scans, Text_data], f)
    i = 0
    return filenames, Digitized_scans, Annotated_scans, Text_data


if __name__ == "__main__":
    setup_tesseract()
    pickle_file = toml.load("config.toml")["pickle"]["likely_us_images"]
    segment(filenames=pickle_file)
