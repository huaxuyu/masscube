import importlib
from importlib.metadata import PackageNotFoundError, version

__all__ = [
    "__version__",
    "process_single_file",
    "untargeted_metabolomics_workflow",
    "run_evaluation",
    "batch_file_processing",
    "read_raw_file_to_obj",
    "generate_sample_table",
    "get_timestamps",
    "build_classifier",
]


try:
    __version__ = version("masscube")
except PackageNotFoundError:
    __version__ = "0+unknown"


_LAZY_EXPORTS = {
    "process_single_file": ".workflows",
    "untargeted_metabolomics_workflow": ".workflows",
    "run_evaluation": ".workflows",
    "batch_file_processing": ".workflows",
    "read_raw_file_to_obj": ".raw_data_utils",
    "generate_sample_table": ".utils_functions",
    "get_timestamps": ".utils_functions",
    "build_classifier": ".classifier_builder",
}


def __getattr__(name):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    mod = importlib.import_module(module_name, __name__)
    attr = getattr(mod, name)
    globals()[name] = attr
    return attr


def __dir__():
    return sorted(set(globals()) | set(__all__))
