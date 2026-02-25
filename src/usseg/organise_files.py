"""Searches a directory and identifies images likely to be doppler ultrasounds"""
# /usr/bin/env python3

import os
import pickle
import re
import toml
import traceback

from concurrent.futures import ThreadPoolExecutor

from usseg import general_functions


def check_file_for_us(file_path):
    """Checks a single file to see if it's a likely ultrasound and returns its path.

    Args:
        file_path: The path to the file to be checked.

    Returns:
        A tuple of (patient_id, file_path) if the file is a likely ultrasound, or `None` otherwise.
    """
    file_lower = file_path.lower()
    
    # Accept all DICOM files
    if file_lower.endswith(('.dcm', '.dicom')):
        # Extract patient ID from the file path
        match = re.search(r"\d{4}", file_path)
        if match:
            patient_id = match.group(0)
            return patient_id, file_path
        return None
    
    # For JPG/PNG files, run scan_type_test
    if file_lower.endswith(('.jpg', '.png')):
        try:
            Fail, df = general_functions.scan_type_test(file_path)
            if Fail == 0:
                # Extract patient ID from the file path
                match = re.search(r"\d{4}", file_path)
                if match:
                    patient_id = match.group(0)
                    return patient_id, file_path
        except Exception:
            traceback.print_exc()
    return None


def get_likely_us(root_dir, pickle_path=None, use_parallel=True):
    """Searches a directory and identifies the images that are likely to be doppler ultrasounds.

    Args:
        root_dir: The path to the directory to be searched.
        pickle_path: The path to the pickle file to save the results to. If `None`, the results will be saved to the current directory.
        use_parallel: Whether to use a parallel thread pool to process the files.

    Returns:
        A list of paths to the images that are likely to be doppler ultrasounds.
    """
    # Initialize a dictionary to store the paths for each patient
    patient_paths = {}

    # Check if the root_dir is a directory or a single file
    if os.path.isdir(root_dir):
        # Collect all JPG, PNG, and DICOM files from the root directory
        all_files = [os.path.join(subdir, file) for subdir, _, files in os.walk(root_dir) for file in files if file.lower().endswith(('.jpg', '.png', '.dcm', '.dicom'))]

        if use_parallel:
            # Using ThreadPoolExecutor to parallelize the file processing
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(check_file_for_us, all_files))
        else:
            results = list(map(check_file_for_us, all_files))

        # Process results and populate patient_paths
        for res in results:
            if res:
                patient_id, file_path = res
                if patient_id in patient_paths:
                    patient_paths[patient_id].append(file_path)
                else:
                    patient_paths[patient_id] = [file_path]

    elif os.path.isfile(root_dir):
        # If it's a single file, directly use the check_file_for_us function
        result = check_file_for_us(root_dir)
        if result:
            patient_id, file_path = result
            patient_paths[patient_id] = [file_path]
    else:
        print(f"{root_dir} is neither a valid directory nor a file.")
        return None

    # save the patient_paths dictionary to a file in current directory
    if pickle_path is not False:
        if pickle_path is None:
            pickle_path = toml.load("config.toml")["pickle"]["likely_us_images"]

        with open(pickle_path, 'wb') as f:
            pickle.dump(patient_paths, f)

    # Convert dictionary values to a list
    all_paths = [path for sublist in patient_paths.values() for path in sublist]

    return all_paths


if __name__ == "__main__":
    config_root_dir = toml.load("config.toml")["root_dir"]
    get_likely_us(config_root_dir)
