"""Initialises the ultrasound-segmentation module.

Makes available the following module:

* general_functions

and the following functions:

* get_likely_us
* data_from_image
* segment (returns five lists; last is mean-wave PNG paths)
* setup_tesseract
* generate_html_from_pkl
* generate_html
* main

Also, sets the attribute '__version__'.
"""
from importlib.metadata import version, PackageNotFoundError
from usseg import general_functions
from usseg.organise_files import get_likely_us
from usseg.single_image_processing import data_from_image
from usseg.segment_files import segment
from usseg.setup_environment import setup_tesseract
from usseg.visualisation_html import generate_html_from_pkl, generate_html
from usseg.main import main
# Replace this with loading from configuration.
# import os
# os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

try:
    __version__ = version("usseg")
except PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0"
