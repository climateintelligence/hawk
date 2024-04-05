import os
import pickle
from typing import Any


def save_to_pkl_file(target_file: str, data: Any, overwrite: bool = True) -> None:
    # Check if the file already exists
    if os.path.exists(target_file) and not overwrite:
        raise ValueError(f"File {target_file} already exists.")

    # Create the directory and parent directories if they don't exist
    os.makedirs(os.path.dirname(target_file), exist_ok=True)

    # Save the data to the file
    with open(target_file, "wb") as f:
        pickle.dump(data, f)


def load_from_pkl_file(source_file: str) -> Any:
    # Check if the file exists
    if not os.path.exists(source_file):
        raise ValueError(f"File {source_file} does not exist.")

    # Load the data from the file
    with open(source_file, "rb") as f:
        data = pickle.load(f)

    return data
