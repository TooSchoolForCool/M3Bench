import os
import json
import shutil
import numpy as np

def mkdir_if_not_exists(dir_name: str, recursive: bool=False) -> None:
    """ Make directory with the given dir_name
    Args:
        dir_name: input directory name that can be a path
        recursive: recursive directory creation
    """
    if os.path.exists(dir_name):
        return 
    
    if recursive:
        os.makedirs(dir_name)
    else:
        os.mkdir(dir_name)

def rmdir_if_exists(dir_name: str, recursive: bool=False) -> None:
    """ Remove directory with the given dir_name
    Args:
        dir_name: input directory name that can be a path
    """
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)

class NumpyArrayEncoder(json.JSONEncoder):
    """
    Python dictionaries can store ndarray array types, but when serialized by dict into JSON files, 
    ndarray types cannot be serialized. In order to read and write numpy arrays, 
    we need to rewrite JSONEncoder's default method. 
    The basic principle is to convert ndarray to list, and then read from list to ndarray.
    """
    def default(self, obj):
        return obj.tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def dict2json(file_name: str, the_dict: dict) -> None:
    """
    Save dict object as json.
    """
    try:
        json_str = json.dumps(the_dict, cls=NumpyArrayEncoder, indent=4)
        with open(file_name, 'w') as json_file:
            json_file.write(json_str)
        return 1
    except:
        return 0 