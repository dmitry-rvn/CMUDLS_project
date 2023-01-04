"""
Utilities module.
"""
from typing import Union
from pathlib import Path
import pickle

import needle as ndl


def save(obj, filepath: Union[str, Path]):
    """
    Save needle object to a file
    (tensor's gradients will be set as None).
    """
    if isinstance(obj, ndl.nn.Module):
        for param in obj.parameters():
            param.grad = None
    elif isinstance(obj, ndl.Tensor):
        obj.grad = None
    with open(filepath, 'wb') as file:
        pickle.dump(obj, file)


def load(filepath: Union[str, Path]):
    """
    Load needle object from a file
    (object will be put on the same device they were before being saved to a file).
    """
    with open(filepath, 'rb') as file:
        return pickle.load(file)
