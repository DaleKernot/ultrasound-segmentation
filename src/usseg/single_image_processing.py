"""Segment a single ultrasound image object.

This module provides a number of functions for segmenting single ultrasound images,
extracting segmentation and textual data from the images.

**Usage:**

To segment a single ultrasound image, you can use the following code:

.. code-block:: python

   from usseg.single_image_processing import data_from_image

   # Load the ultrasound image.
   PIL_img = ...
   cv2_img = ...

   # Extract segmentation and textual data from the image.
   df, XYdata = data_from_image(PIL_img, cv2_img)

"""
# Python imports
import logging
import os
from PIL import Image

# Module imports
import matplotlib.pyplot as plt
import cv2
import pandas as pd

# Local imports
from usseg import general_functions

logger = logging.getLogger(__file__)


def data_from_image(pil_img=None, cv2_img=None, image_path=None):
    """Extract segmentation and textual data from an image.

    Transitional API (temporary):
        This function is migrating from accepting pre-loaded images
        (pil_img, cv2_img) to accepting an image file path instead.

        Old usage (still supported temporarily):
            data_from_image(PIL_image, cv2_image)

        New usage (preferred):
            data_from_image(image_path="path/to/image")

        Legacy arguments will be removed in a future update once
        dependent codes have migrated.

    Args:
        pil_img (Pillow Image object) : The image in Pillow format.
        cv2_img (cv2 Image object) : The image in cv2 format.

        image_path (str, optional) : Path to the image file. Transitional option.
            If provided, this function will judge file type and load pil_img and cv2_img internally.

    Returns:
        df (pandas dataframe) : Dataframe of extracted text.
        XYdata (list) : X and Y coordinates of the extracted segmentation.
    """

    # Guard invalid input combinations
    if image_path is not None and (pil_img is not None or cv2_img is not None):
        raise ValueError(
            "Provide either image_path OR (pil_img AND cv2_img), not both."
        )


    if image_path is not None:
        # Classify file type for downstream handling (image vs DICOM)
        ext = os.path.splitext(image_path)[1].lower()
        us_dicom = ext in (".dcm", ".dicom")
        # is_image: jpeg, png, or other common image formats
        us_image = ext in (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp")
        if us_image: #Load image if image path
            pil_img = Image.open(image_path)
            cv2_img = cv2.imread(image_path)
        elif us_dicom: #Load DICOM if image path is a DICOM file
            dicom_metadata = general_functions.extract_dicom_metadata(image_path)
            PIL_image, cv2_img = general_functions.extract_doppler_from_dicom(image_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    else:
        # Guard invalid input combinations and allow legacy inputs
        if pil_img is not None and cv2_img is not None:
                us_image = True
        else:
            raise ValueError(
                "Provide either image_path OR (pil_img AND cv2_img), not both."
                )

    # Downstream pipeline aligned with segment_files: branch on us_image vs us_dicom
    if us_image:
        # --- Image path: text extraction, initial segmentation, axis search, single-axis digitisation ---
        PIL_image_RGB = pil_img.convert("RGB")
        COL = general_functions.colour_extract_vectorized(PIL_image_RGB, [255, 255, 100], 95, 95)
        text_extract_failed, df = general_functions.text_from_greyscale(cv2_img, COL)
        if text_extract_failed:
            logger.warning("Couldn't extract text from image. Continuing...")
        else:
            logger.info("Completed colour extraction.")

        # No error handling for initial segmentation as impossible to complete segmentation
        # without segmentation mask.
        segmentation_mask, Xmin, Xmax, Ymin, Ymax = general_functions.initial_segmentation(
            input_image_obj=PIL_image_RGB
        )
        Left_dimensions, Right_dimensions = general_functions.define_end_rois(
            segmentation_mask, Xmin, Xmax, Ymin, Ymax
        )

        
        # Initialise axis containers so the functions can pass if one side fails.
        Lnumber = None
        Lpositions = None
        Rnumber = None
        Rpositions = None 

        try:
            # Search for ticks and labels - Left axis
            (
                Cs, ROIAX, CenPoints, onY, BCs, TYLshift, thresholded_image, Side,
                Left_dimensions, Right_dimensions, ROI2, ROI3,
            ) = general_functions.search_for_ticks(
                cv2_img, "Left", Left_dimensions, Right_dimensions
            )
            ROIAX, Lnumber, Lpositions, ROIL = general_functions.search_for_labels(
                Cs, ROIAX, CenPoints, onY, BCs, TYLshift, Side,
                Left_dimensions, Right_dimensions, cv2_img, ROI2, ROI3,
            )
            # Validate and lightly clean left-axis ticks/labels
            Lnumber, Lpositions = general_functions.validate_axis_ticks(
                Lnumber, Lpositions, side="Left"
            )
        except Exception:
            logger.exception("Single-image: failed Left axes search")

        try:
            # Search for ticks and labels - Right axis
            (
                Cs, ROIAX, CenPoints, onY, BCs, TYLshift, thresholded_image, Side,
                Left_dimensions, Right_dimensions, ROI2, ROI3,
            ) = general_functions.search_for_ticks(
                cv2_img, "Right", Left_dimensions, Right_dimensions
            )
            ROIAX, Rnumber, Rpositions, ROIR = general_functions.search_for_labels(
                Cs, ROIAX, CenPoints, onY, BCs, TYLshift, Side,
                Left_dimensions, Right_dimensions, cv2_img, ROI2, ROI3,
            )
            # Validate and lightly clean right-axis ticks/labels
            Rnumber, Rpositions = general_functions.validate_axis_ticks(
                Rnumber, Rpositions, side="Right"
            )
        except Exception:
            logger.exception("Single-image: failed Right axes search")

        try:
            y_zero = general_functions.estimate_zero_line_y(
                left_numbers=Lnumber, left_positions=Lpositions,
                right_numbers=Rnumber, right_positions=Rpositions,
            )
            if y_zero is not None:
                logger.info(f"Single-image: estimated y=0 line at y={y_zero:.2f}")
        except Exception:
            logger.exception("Single-image: zero-line estimation failed")
            y_zero = None

        (
            refined_segmentation_mask,
            top_curve_mask,
            top_curve_coords,
        ) = general_functions.segment_refinement(
            cv2_img, Xmin, Xmax, Ymin, Ymax, y_zero=y_zero
        )
        Xplot, Yplot, Ynought = general_functions.plot_digitized_data_single_axis(
            Rnumber, Rpositions, Lnumber, Lpositions, top_curve_coords,
        )

        if not text_extract_failed:
            try:
                df = general_functions.plot_correction(Xplot, Yplot, df)
            except Exception:
                logger.exception("Single-image: plot_correction failed")

        plt.close("all")
        return df, [Xplot, Yplot]

    elif us_dicom:
        # --- DICOM path: metadata dimensions, label text, no axis search, DICOM digitisation + metrics ---
        label_result = general_functions.extract_dicom_label_text(cv2_img)
        Xmin = dicom_metadata.get("RegionLocationMinX0")
        Xmax = dicom_metadata.get("RegionLocationMaxX1")
        Ymin = dicom_metadata.get("RegionLocationMinY0")
        Ymax = dicom_metadata.get("RegionLocationMaxY1")
        y_zero = (
            dicom_metadata.get("RegionLocationMinY0", 0)
            + dicom_metadata.get("ReferencePixelY0", 0)
        )

        (
            refined_segmentation_mask,
            top_curve_mask,
            top_curve_coords,
        ) = general_functions.segment_refinement(
            cv2_img, Xmin, Xmax, Ymin, Ymax, y_zero=y_zero
        )
        Xplot, Yplot, Ynought = general_functions.plot_digitized_data_dicom(
            dicom_metadata, top_curve_coords=top_curve_coords
        )
        df = general_functions.waveform_metrics_from_digitized(Xplot, Yplot)

        if label_result and not df.empty:
            label_str = " ".join(
                filter(None, [label_result.get("side"), label_result.get("vessel")])
            ).strip()
            if label_str:
                label_row = pd.DataFrame(
                    [{"Line": 0, "Word": "Label", "Value": label_str, "Unit": "", "Digitized Value": ""}],
                    columns=df.columns,
                )
                df = pd.concat([label_row, df], ignore_index=True)
            df["Line"] = range(1, len(df) + 1)

        plt.close("all")
        return df, [Xplot, Yplot]

    else:
        raise ValueError("Unsupported file type.")
