from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from masscube import workflows
from masscube.feature_detection import Feature, detect_features
from masscube.feature_evaluation import calculate_noise_score
from masscube.params import Params
from masscube.raw_data_utils import MSData, Scan


def _ms1_scan(scan_time, signals):
    return Scan(
        raw_file_id=int(scan_time),
        level=1,
        scan_time=float(scan_time),
        signals=np.asarray(signals, dtype=np.float32).reshape(-1, 2),
    )


def test_empty_ms1_scans_advance_feature_gap_counter():
    data = MSData()
    data.params = Params()
    data.params.feature_gap_tol = 1
    data.scans = [
        _ms1_scan(0, [[100.0, 1000.0]]),
        _ms1_scan(1, []),
        _ms1_scan(2, []),
        _ms1_scan(3, [[100.0, 900.0]]),
    ]
    data.ms1_idx_arr = np.arange(4, dtype=np.int32)

    features = detect_features(data)

    assert len(features) == 2


def test_noise_score_does_not_modify_input_signal():
    signal = np.array([100.0, 1.0, 80.0], dtype=np.float32)
    original = signal.copy()

    calculate_noise_score(signal, rel_int_tol=0.05, len_tol=1)

    np.testing.assert_array_equal(signal, original)


def test_ms2_allocation_uses_expanded_feature_peak_edges():
    data = MSData()
    feature = Feature()
    feature.mz = 100.0
    feature.rt_seq = [1.0, 2.0]
    feature.peak_edges = (0.8, 2.2)
    feature.peak_height = 1000
    data.features = [feature]
    data.feature_mz_arr = np.array([feature.mz])

    ms2 = Scan(
        raw_file_id=1,
        level=2,
        scan_time=0.9,
        signals=np.array([[50.0, 100.0]], dtype=np.float32),
        precursor_mz=100.05,
    )
    data.scans = [ms2]
    data.ms2_idx_arr = np.array([0], dtype=np.int32)

    data.allocate_ms2_to_features()

    assert feature.ms2_seq == [ms2]


def test_process_single_file_marks_empty_ms1_as_skipped(monkeypatch):
    data = MSData()
    data.params = Params()
    data.ms1_idx_arr = np.empty(0, dtype=np.int32)

    monkeypatch.setattr(
        workflows,
        "find_raw_data_info",
        lambda _: ("qtof", "positive", True, None),
    )
    monkeypatch.setattr(workflows, "get_raw_data_format", lambda _: "mzml")
    monkeypatch.setattr(workflows, "read_raw_file_to_obj", lambda *_args, **_kwargs: data)

    with pytest.raises(
        workflows.SingleFileProcessingSkipped, match="No valid MS1 data"
    ):
        workflows.process_single_file("empty.mzML", return_data=False)


def test_process_single_file_reraises_processing_errors(monkeypatch):
    def fail(_):
        raise ValueError("invalid raw data")

    monkeypatch.setattr(workflows, "find_raw_data_info", fail)

    with pytest.raises(ValueError, match="invalid raw data"):
        workflows.process_single_file(Path("invalid.mzML"))


def test_process_single_file_resolves_options_from_params(tmp_path, monkeypatch):
    library_path = tmp_path / "library.pkl"
    library_path.touch()

    params = Params()
    params.segment_features = False
    params.group_features_single_file = True
    params.rt_tol_feature_grouping = 0.123
    params.annotate_ms2 = True
    params.ms2_library_path = str(library_path)
    params.fuzzy_search = False
    params.consider_rt = True

    calls = {"detect": 0, "segment": 0, "finalize": 0}

    class Data:
        def __init__(self):
            self.params = params
            self.metadata = SimpleNamespace(is_centroid=True, file_name="sample")
            self.ms1_idx_arr = np.array([0], dtype=np.int32)

        def detect_features(self):
            calls["detect"] += 1

        def segment_features(self):
            calls["segment"] += 1

        def finalize_features(self):
            calls["finalize"] += 1

    data = Data()
    annotation_options = {}
    grouping_options = {}

    def read_data(*_args, **kwargs):
        data.params = kwargs["params"]
        assert kwargs["ms_info"] == ("qtof", "positive", True, None)
        return data

    monkeypatch.setattr(
        workflows,
        "find_raw_data_info",
        lambda _: ("qtof", "positive", True, None),
    )
    monkeypatch.setattr(workflows, "get_raw_data_format", lambda _: "mzml")
    monkeypatch.setattr(workflows, "read_raw_file_to_obj", read_data)
    monkeypatch.setattr(
        workflows,
        "annotate_features",
        lambda **kwargs: annotation_options.update(kwargs),
    )
    monkeypatch.setattr(
        workflows,
        "group_features_single_file",
        lambda d, rt_tol: grouping_options.update({"data": d, "rt_tol": rt_tol}),
    )

    result = workflows.process_single_file("sample.mzML", params=params)

    assert result is data
    assert data.params is not params
    assert calls == {"detect": 1, "segment": 0, "finalize": 1}
    assert annotation_options["fuzzy_search"] is False
    assert annotation_options["consider_rt"] is True
    assert annotation_options["similarity_method"] == "unweighted_entropy"
    assert grouping_options == {"data": data, "rt_tol": 0.123}


def test_process_single_file_applies_overrides_before_validation(monkeypatch):
    params = Params()
    params.segment_features = "invalid"

    class Data:
        def __init__(self):
            self.params = None
            self.metadata = SimpleNamespace(is_centroid=True, file_name="sample")
            self.ms1_idx_arr = np.array([0], dtype=np.int32)

        def detect_features(self):
            pass

        def finalize_features(self):
            pass

    data = Data()
    monkeypatch.setattr(
        workflows,
        "find_raw_data_info",
        lambda _: ("qtof", "positive", True, None),
    )
    monkeypatch.setattr(workflows, "get_raw_data_format", lambda _: "mzml")

    def read_data(*_args, **kwargs):
        data.params = kwargs["params"]
        return data

    monkeypatch.setattr(workflows, "read_raw_file_to_obj", read_data)

    workflows.process_single_file(
        "sample.mzML", params=params, segment_feature=False
    )

    assert data.params.segment_features is False
    assert params.segment_features == "invalid"


def test_process_single_file_rejects_non_boolean_overrides(monkeypatch):
    monkeypatch.setattr(
        workflows,
        "find_raw_data_info",
        lambda _: ("qtof", "positive", True, None),
    )

    with pytest.raises(TypeError, match="segment_feature"):
        workflows.process_single_file("sample.mzML", segment_feature="false")


def test_process_single_file_deep_copies_params(monkeypatch):
    params = Params()
    params.extra_options = {"labels": []}

    class Data:
        def __init__(self):
            self.params = None
            self.metadata = SimpleNamespace(is_centroid=True, file_name="sample")
            self.ms1_idx_arr = np.array([0], dtype=np.int32)

        def detect_features(self):
            self.params.extra_options["labels"].append("processed")

        def segment_features(self):
            pass

        def finalize_features(self):
            pass

    data = Data()
    monkeypatch.setattr(
        workflows,
        "find_raw_data_info",
        lambda _: ("qtof", "positive", True, None),
    )
    monkeypatch.setattr(workflows, "get_raw_data_format", lambda _: "mzml")

    def read_data(*_args, **kwargs):
        data.params = kwargs["params"]
        return data

    monkeypatch.setattr(workflows, "read_raw_file_to_obj", read_data)

    workflows.process_single_file("sample.mzML", params=params)

    assert params.extra_options == {"labels": []}
    assert data.params.extra_options == {"labels": ["processed"]}


def test_process_single_file_uses_output_flags_and_explicit_destinations(
    tmp_path, monkeypatch
):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_path = source_dir / "sample.mzML"
    source_path.touch()
    explicit_table_dir = tmp_path / "tables"
    explicit_bpc_dir = tmp_path / "bpcs"

    params = Params()
    params.output_single_file = False
    params.plot_bpc = False
    calls = {}

    class Data:
        def __init__(self):
            self.params = None
            self.metadata = SimpleNamespace(is_centroid=True, file_name="sample")
            self.ms1_idx_arr = np.array([0], dtype=np.int32)

        def detect_features(self):
            pass

        def segment_features(self):
            pass

        def finalize_features(self):
            pass

        def output_single_file(self, path):
            calls["table"] = path

        def plot_bpc(self, output_dir):
            calls["bpc"] = output_dir

    data = Data()
    monkeypatch.setattr(
        workflows,
        "find_raw_data_info",
        lambda _: ("qtof", "positive", True, None),
    )
    monkeypatch.setattr(workflows, "get_raw_data_format", lambda _: "mzml")

    def read_data(*_args, **kwargs):
        data.params = kwargs["params"]
        return data

    monkeypatch.setattr(workflows, "read_raw_file_to_obj", read_data)

    workflows.process_single_file(
        source_path,
        params=params,
        output_dir=explicit_table_dir,
        bpc_output_dir=explicit_bpc_dir,
    )

    assert calls == {
        "table": str(explicit_table_dir / "sample.txt"),
        "bpc": str(explicit_bpc_dir / "sample_bpc.png"),
    }

    calls.clear()
    params.output_single_file = True
    params.plot_bpc = True
    workflows.process_single_file(source_path, params=params)
    assert calls == {
        "table": str(source_dir / "sample.txt"),
        "bpc": str(source_dir / "sample_bpc.png"),
    }


def test_process_single_file_only_suppresses_library_errors(monkeypatch):
    params = Params()
    params.annotate_ms2 = True
    params.ms2_library_path = "library.pkl"

    class Data:
        def __init__(self):
            self.params = None
            self.metadata = SimpleNamespace(is_centroid=True, file_name="sample")
            self.ms1_idx_arr = np.array([0], dtype=np.int32)

        def detect_features(self):
            pass

        def segment_features(self):
            pass

        def finalize_features(self):
            pass

    data = Data()
    monkeypatch.setattr(
        workflows,
        "find_raw_data_info",
        lambda _: ("qtof", "positive", True, None),
    )
    monkeypatch.setattr(workflows, "get_raw_data_format", lambda _: "mzml")

    def read_data(*_args, **kwargs):
        data.params = kwargs["params"]
        return data

    monkeypatch.setattr(workflows, "read_raw_file_to_obj", read_data)
    monkeypatch.setattr(
        workflows,
        "annotate_features",
        lambda **_kwargs: (_ for _ in ()).throw(
            workflows.MS2LibraryError("invalid library")
        ),
    )
    assert workflows.process_single_file("sample.mzML", params=params) is data

    monkeypatch.setattr(
        workflows,
        "annotate_features",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("algorithm bug")),
    )
    with pytest.raises(RuntimeError, match="algorithm bug"):
        workflows.process_single_file("sample.mzML", params=params)


def test_project_single_file_worker_reports_expected_skips(monkeypatch):
    monkeypatch.setattr(
        workflows,
        "process_single_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            workflows.SingleFileProcessingSkipped("no selected MS1 scans")
        ),
    )

    result = workflows._process_project_single_file(
        "sample.mcraw", Params(), "tables", None
    )

    assert result["sample_name"] == "sample"
    assert result["status"] == "skipped"
    assert result["reason"] == "no selected MS1 scans"
    assert result["feature_count"] is None

    monkeypatch.setattr(
        workflows,
        "process_single_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("algorithm bug")
        ),
    )
    result = workflows._process_project_single_file(
        "sample.mcraw", Params(), "tables", None
    )
    assert result["status"] == "failed"
    assert result["reason"] == "RuntimeError: algorithm bug"


def test_params_do_not_store_single_file_metadata():
    params = Params()

    for name in (
        "file_name",
        "file_path",
        "file_format",
        "ms_type",
        "ion_mode",
        "is_centroid",
        "scan_time_unit",
    ):
        assert not hasattr(params, name)

    for name in (
        "sample_metadata",
        "project_dir",
        "sample_dir",
        "single_file_dir",
        "tmp_file_dir",
        "ms2_matching_dir",
        "bpc_dir",
        "project_file_dir",
        "normalization_dir",
        "statistics_dir",
        "problematic_files",
    ):
        assert not hasattr(params, name)


def test_parameter_csv_rejects_metadata_and_parses_processing_flags(tmp_path):
    invalid_path = tmp_path / "invalid.csv"
    invalid_path.write_text("name,value\nion_mode,positive\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-parameter field"):
        Params().read_parameters_from_csv(invalid_path)

    compatibility_params = Params()
    compatibility_params.read_parameters_from_csv(
        invalid_path, ignore_metadata_fields=True
    )
    assert not hasattr(compatibility_params, "ion_mode")

    valid_path = tmp_path / "valid.csv"
    valid_path.write_text(
        "name,value\nsegment_features,no\nannotate_ms2,yes\n"
        "group_features_single_file,true\nrt_tol_feature_grouping,0.12\n",
        encoding="utf-8",
    )
    params = Params()
    params.read_parameters_from_csv(valid_path)

    assert params.segment_features is False
    assert params.annotate_ms2 is True
    assert params.group_features_single_file is True
    assert params.rt_tol_feature_grouping == pytest.approx(0.12)


def test_parameter_validation_fails_fast_for_invalid_values():
    params = Params()
    params.percent_cpu_to_use = 1.2

    with pytest.raises(ValueError, match="percent_cpu_to_use"):
        params.check_parameters()
