"""Workflow-level cache management for raw mass spectrometry data."""

import os

from ..params import Params
from .io import find_raw_data_info, get_raw_data_format
from .mcraw import ensure_mcraw, mcraw_matches_source


def _mcraw_cache_path(file_name, tmp_file_dir):
    """Return the workflow cache path for a raw data source."""

    source = os.path.abspath(os.fspath(file_name))
    return os.path.join(
        os.fspath(tmp_file_dir),
        os.path.splitext(os.path.basename(source))[0] + ".mcraw",
    )


def _prepare_mcraw_cache(file_name, tmp_file_dir, params=None, ms_info=None):
    """Return the mcraw analysis source for one raw file.

    Existing mcraw inputs are used directly. mzML and Bruker ``.d`` inputs are
    decoded once into the workflow's tmp directory; a source fingerprint makes
    later calls a cheap validity check.
    """

    source = os.path.abspath(os.fspath(file_name))
    if get_raw_data_format(source) == "mcraw":
        return source

    if ms_info is None:
        ms_info = find_raw_data_info(source)
    ms_type, ion_mode, is_centroid, _ = ms_info
    if not is_centroid:
        raise ValueError(f"File is not centroid and cannot be cached: {source}")

    cache_params = params
    if cache_params is None:
        cache_params = Params()
        cache_params.set_default(ms_type, ion_mode)

    os.makedirs(tmp_file_dir, exist_ok=True)
    output_path = _mcraw_cache_path(source, tmp_file_dir)
    return os.fspath(
        ensure_mcraw(source, output_path, cache_params, ms_info=ms_info)
    )


def _mcraw_cache_is_current(file_name, tmp_file_dir):
    """Return whether a raw source already has a valid workflow cache."""

    source = os.path.abspath(os.fspath(file_name))
    if get_raw_data_format(source) == "mcraw":
        return True
    return mcraw_matches_source(
        _mcraw_cache_path(source, tmp_file_dir),
        source,
    )
