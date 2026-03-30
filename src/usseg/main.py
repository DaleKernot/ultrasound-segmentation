"""
Main script for running the file organisation and segmentation.

This script can be used to segment ultrasound images and generate an output.html file containing the segmented output.

**Usage:**

python usseg.main.py root_dir

where `root_dir` is the root directory containing the ultrasound images to be segmented.

**Example:**

To segment all of the ultrasound images in the directory `E:/us-data-anon`, you would run the following command:

python usseg.main.py E:/us-data-anon

This will generate an output.html file in the current directory containing the segmented output for all of the ultrasound images in the `E:/us-data-anon` directory.

**Configuration options:**

The script can be configured using the `config.toml` file. The `config.toml` file should be placed in the same directory as the script.

The only configuration option currently supported is the `root_dir` key. The `root_dir` key specifies the root directory containing the ultrasound images to be segmented.

**Known issues and limitations:**

The script is currently under development and may contain bugs.

"""

#/usr/bin/env python3

# Python imports
import cProfile
import pstats
import io
from pstats import SortKey

# Module imports
import toml
import logging

# Local imports
import usseg


def prof(fn, *args, **kwargs):

    pr = cProfile.Profile()

    pr.enable()
    rtn_val = fn(*args, **kwargs)
    pr.disable()

    s = io.StringIO()
    sort_by = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sort_by)
    ps.print_stats()

    with open(f"{fn.__name__}.log", "w+") as f:
        f.write(s.getvalue())

    return rtn_val


def main(root_dir):
    """Main function that performs all of the segmentation on a root directory"""

    # Checks and sets up the tesseract environment
    usseg.setup_tesseract()

    # Gets a list of likely ultrasound images from root dir and saves them to a pickle file.
    filenames = prof(usseg.get_likely_us, root_dir)

    # Segments and digitises the pre-selected ultrasound images.
    # filenames = "Path/to/a/single/test/file.JPG"
    prof(usseg.segment, filenames)

    # Generates an output.html of the segmented output
    prof(usseg.generate_html_from_pkl)


if __name__ == "__main__":
    # Ensure INFO-level logs (including general_functions debug) are visible
    # and persist to a local log file when running main.py directly.
    logging.basicConfig(
        level=logging.INFO,
        filename="batch_main.log",
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    config_root_dir = toml.load("config.toml")["root_dir"]
    # root_dir = "Path/to/a/folder/of/images"
    main(config_root_dir)
