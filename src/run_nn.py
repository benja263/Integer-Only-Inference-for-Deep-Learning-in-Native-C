
"""
Module interfacing between the C files and python
"""
import ctypes
import ctypes.util
import sys
import os

import numpy as np


def load_c_lib():
    """
    Load C shared library
    :return:
    """
    try:
        c_lib = ctypes.CDLL(f"{os.path.dirname(os.path.abspath(__file__))}/nn.so")
    except OSError:
        print("Unable to load the requested C library")
        sys.exit()
    return c_lib

def ensure_contiguous(array):
    """
    Ensure that array is contiguous
    :param array:
    :return:
    """
    return np.ascontiguousarray(array) if not array.flags['C_CONTIGUOUS'] else array


def run_mlp(x, c_lib):
    """
    Call 'run_mlp' function from C in Python
    :param x:
    :param c_lib:
    :return:
    """
    N = len(x)
    x = x.flatten()
    x = ensure_contiguous(x.numpy())
    x = x.astype(np.intc)
    # print(x)
    class_indices = ensure_contiguous(np.zeros(N, dtype=np.uintc))

    c_int_p = ctypes.POINTER(ctypes.c_int)
    c_uint_p = ctypes.POINTER(ctypes.c_uint)

    c_run_mlp = c_lib.run_mlp
    c_run_mlp.argtypes = (c_int_p, ctypes.c_uint, c_uint_p)
    c_run_mlp.restype = None
    c_run_mlp(x.ctypes.data_as(c_int_p), ctypes.c_uint(N), 
              class_indices.ctypes.data_as(c_uint_p)
                      )

    return np.ctypeslib.as_array(class_indices, N)


