"""Test the single image processing function."""

# Python imports
import logging

# Module imports
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Local imports
from usseg import data_from_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


def test_data_from_image():
    """Test the data_from_image function."""
    img_path = "C:/Users/user/OneDrive/Desktop/STP/Master's/Year 3/data files/database_test/EYE_455_29+2_right_ophthalmic_repeat.jpg"

    #PIL_image = Image.open(img_path)
    #cv2_image = np.array(PIL_image)
    #logger.info(f"Loaded image with shape {cv2_image.shape} and type {cv2_image.dtype}")

    df, (xdata, ydata) = data_from_image(image_path=img_path)

    fig, ax_trace = plt.subplots(figsize=(8, 5))
    ax_trace.plot(xdata, ydata, "-")
    ax_trace.set_title("Extracted trace")
    ax_trace.set_xlabel("Time")
    ax_trace.set_ylabel("Velocity")
    ax_trace.grid(True, alpha=0.3)
    fig.tight_layout()

    # Makes sure that the lists aren't empty
    assert xdata
    assert ydata

    logger.info(f"Extracted the following text from {img_path}:\n{df}")

if __name__ == "__main__":
    test_data_from_image()
    plt.show()
    logger.info(f"{__file__} tests have passed!")
