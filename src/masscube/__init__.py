# from .workflows import process_single_file, untargeted_metabolomics_workflow, run_evaluation, batch_file_processing
# from .raw_data_utils import read_raw_file_to_obj
# from .utils_functions import generate_sample_table, get_timestamps
# from .classifier_builder import build_classifier

import importlib

__all__ = [
    "process_single_file",
    "untargeted_metabolomics_workflow",
    "run_evaluation",
    "batch_file_processing",
    "read_raw_file_to_obj",
    "generate_sample_table",
    "get_timestamps",
    "build_classifier",
]

def __getattr__(name):
    if name in {"process_single_file", "untargeted_metabolomics_workflow", "run_evaluation", "batch_file_processing"}:
        mod = importlib.import_module(".workflows", __name__)
        return getattr(mod, name)

    if name == "read_raw_file_to_obj":
        mod = importlib.import_module(".raw_data_utils", __name__)
        return getattr(mod, name)

    if name in {"generate_sample_table", "get_timestamps"}:
        mod = importlib.import_module(".utils_functions", __name__)
        return getattr(mod, name)

    if name == "build_classifier":
        mod = importlib.import_module(".classifier_builder", __name__)
        return getattr(mod, name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")