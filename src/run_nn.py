
"""
Module interfacing between the C files and python
"""
import ctypes
import ctypes.util
import sys
import os

import numpy as np


def load_c_lib(library):
    """
    Load C shared library
    :return:
    """
    try:
        c_lib = ctypes.CDLL(f"{os.path.dirname(os.path.abspath(__file__))}/{library}")
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


def run_convnet(x, c_lib):
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

    c_run_convnet = c_lib.run_convnet
    # c_run_convnet.argtypes = (c_int_p, ctypes.c_uint, c_uint_p)
    c_run_convnet.argtypes = (c_int_p, c_uint_p)
    c_run_convnet.restype = None
    c_run_convnet(x.ctypes.data_as(c_int_p),
              class_indices.ctypes.data_as(c_uint_p)
    # c_run_convnet(x.ctypes.data_as(c_int_p), ctypes.c_uint(N), 
    #           class_indices.ctypes.data_as(c_uint_p)
                      )

    return np.ctypeslib.as_array(class_indices, N)


def run_convnetf(x, c_lib):
    """
    Call 'run_mlp' function from C in Python
    :param x:
    :param c_lib:
    :return:
    """
    N = len(x)
    x = x.flatten()
    x = ensure_contiguous(x.numpy())
    x = x.astype(np.single)
    # print(x)
    class_indices = ensure_contiguous(np.zeros(N, dtype=np.uintc))

    c_float_p = ctypes.POINTER(ctypes.c_float)
    c_uint_p = ctypes.POINTER(ctypes.c_uint)

    c_run_convnetf = c_lib.run_convnetf
    # c_run_convnetf.argtypes = (c_float_p, ctypes.c_uint, c_uint_p)
    c_run_convnetf.argtypes = (c_float_p,c_uint_p)
    c_run_convnetf.restype = None
    # c_run_convnetf(x.ctypes.data_as(c_float_p), ctypes.c_uint(N), 
    #           class_indices.ctypes.data_as(c_uint_p)
    c_run_convnetf(x.ctypes.data_as(c_float_p),
              class_indices.ctypes.data_as(c_uint_p)
                      )

    return np.ctypeslib.as_array(class_indices, N)
    



