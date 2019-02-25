"""Helpers and utility functions for handling system I/O."""
import os
import pickle as pk


def save_result(sample, filename, file_addr):
    """Saves result to target address."""
    # check file address
    if not filename.endswith(".pk"):
        filename = "{}.pk".format(filename)

    # create full file name path
    file_path_full = os.path.join(file_addr, filename)
    file_path_parent = os.path.dirname(file_path_full)

    os.makedirs(file_path_parent, exist_ok=True)

    # save
    with open(file_path_full, 'wb') as file:
        pk.dump(sample, file, protocol=pk.HIGHEST_PROTOCOL)


def load_results(filename_list, file_addr):
    """Loads files from the filename_list.

    Args:
        filename_list: (list of str) List of file names to load.
        file_addr: (str) parent directory for files.

    Returns:
        (dict) A dictionary of file names and the loaded object.
    """
    file_dict = dict()

    for filename in filename_list:
        file_path_full = os.path.join(file_addr, filename)

        with open(file_path_full, 'rb') as file:
            file_dict[filename] = pk.load(file)

    return file_dict
