"""Raw MS data models and format-specific readers.

Public imports from the former ``masscube.raw_data_utils`` module remain
valid. Imports are lazy so the standalone TDF decoder in :mod:`._tdf` retains
its standard-library-only runtime boundary.
"""

import importlib


__all__ = [
    "CalibratedTimsFrame",
    "CorruptTDFError",
    "FrameMetadata",
    "MSData",
    "McrawStore",
    "PasefWindow",
    "RawTimsFrame",
    "Scan",
    "SingleFileMetadata",
    "TDFError",
    "TDFReader",
    "UNSUPPORTED_RAW_FORMAT_MESSAGE",
    "UnsupportedTDFError",
    "ZstdUnavailableError",
    "cal_precursor_ion_fraction",
    "default_processing_status",
    "default_single_file_metadata",
    "find_best_ms2",
    "find_raw_data_info",
    "get_raw_data_format",
    "ensure_mcraw",
    "inspect_bruker_d",
    "inspect_mcraw",
    "is_supported_raw_data_path",
    "load_mcraw",
    "mcraw_matches_source",
    "read_mcraw_manifest",
    "read_raw_file_to_obj",
    "read_tims_d_to_msdata",
    "save_mcraw",
]


_LAZY_EXPORTS = {
    "CalibratedTimsFrame": "._tdf",
    "CorruptTDFError": "._tdf",
    "FrameMetadata": "._tdf",
    "PasefWindow": "._tdf",
    "RawTimsFrame": "._tdf",
    "TDFError": "._tdf",
    "TDFReader": "._tdf",
    "UnsupportedTDFError": "._tdf",
    "ZstdUnavailableError": "._tdf",
    "MSData": ".core",
    "McrawStore": ".mcraw",
    "Scan": ".core",
    "SingleFileMetadata": ".core",
    "UNSUPPORTED_RAW_FORMAT_MESSAGE": ".core",
    "cal_precursor_ion_fraction": ".core",
    "default_processing_status": ".core",
    "default_single_file_metadata": ".core",
    "find_best_ms2": ".core",
    "find_raw_data_info": ".io",
    "get_raw_data_format": ".io",
    "ensure_mcraw": ".mcraw",
    "is_supported_raw_data_path": ".io",
    "inspect_mcraw": ".mcraw",
    "load_mcraw": ".mcraw",
    "mcraw_matches_source": ".mcraw",
    "read_mcraw_manifest": ".mcraw",
    "read_raw_file_to_obj": ".io",
    "inspect_bruker_d": ".bruker",
    "read_tims_d_to_msdata": ".bruker",
    "save_mcraw": ".mcraw",
}


def __getattr__(name):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    module = importlib.import_module(module_name, __name__)
    attribute = getattr(module, name)
    globals()[name] = attribute
    return attribute


def __dir__():
    return sorted(set(globals()) | set(__all__))
