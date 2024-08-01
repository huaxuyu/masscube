"""
msjson.py - JSON utilities for MassCube

This module defines a new data format for mass spectrometry data, msjson.

It provides:

1. Structure of the msjson format.
2. Convert the mzML or mzXML file to msjson format.
3. Convert the MSData object to msjson format and vice versa.

msjson design philosophy:

1. smaller file size.
2. faster loading and parsing in Python.
3. human and machine readable.
"""

import json
import os

from .raw_data_utils import MSData