from types import SimpleNamespace

import numpy as np
import pandas as pd

from masscube.params import Params
from masscube.raw_data_utils import (
    MSData,
    McrawStore,
    Scan,
    load_mcraw,
    mcraw_matches_source,
    read_mcraw_manifest,
    save_mcraw,
)


def _make_mcraw_data():
    data = MSData()
    data.update_metadata(
        {
            "file_path": "/raw/sample.mzML",
            "ms_type": "qtof",
            "ion_mode": "positive",
            "is_centroid": True,
            "file_format": "mzml",
        }
    )
    data.params = Params()
    data.params.set_default("qtof", "positive")
    data.scans = [
        Scan(
            file_name="sample",
            raw_file_id=10,
            level=1,
            scan_time=0.95,
            signals=np.array([[100.0, 5.0], [200.0, 2.0]], dtype=np.float32),
        ),
        Scan(
            file_name="sample",
            raw_file_id=11,
            level=2,
            scan_time=0.97,
            signals=np.array([[50.0, 3.0]], dtype=np.float32),
            precursor_mz=100.0,
            isolation_window=[0.5, 0.5],
        ),
        Scan(
            file_name="sample",
            raw_file_id=12,
            level=1,
            scan_time=1.00,
            signals=np.array([[100.0, 10.0]], dtype=np.float32),
        ),
        Scan(
            file_name="sample",
            raw_file_id=13,
            level=2,
            scan_time=1.02,
            signals=np.array([[60.0, 4.0]], dtype=np.float32),
            precursor_mz=200.0,
            isolation_window=[0.5, 0.5],
        ),
    ]
    for scan in data.scans:
        scan.collision_energy = 20.0 if scan.level == 2 else None
    return data


def test_mcraw_round_trip_uses_common_and_level_metadata(tmp_path):
    data = _make_mcraw_data()
    path = tmp_path / "sample.mcraw"
    save_mcraw(data, path)

    manifest = read_mcraw_manifest(path)
    assert "scan_table.npy" not in {item.name for item in path.iterdir()}
    assert manifest["common_scan_metadata"]["file_name"] == "sample"
    assert manifest["level_metadata"]["1"]["collision_energy"] == {
        "kind": "float",
        "storage": "constant",
        "value": None,
    }
    assert manifest["level_metadata"]["2"]["collision_energy"] == {
        "kind": "float",
        "storage": "constant",
        "value": 20.0,
    }
    assert manifest["level_metadata"]["2"]["precursor_mz"]["storage"] == "array"

    restored = load_mcraw(path, preprocess=False)
    assert restored.metadata.file_name == "sample"
    assert restored.ms1_idx_arr.tolist() == [0, 2]
    assert restored.ms2_idx_arr.tolist() == [1, 3]
    for expected, actual in zip(data.scans, restored.scans):
        assert actual.raw_file_id == expected.raw_file_id
        assert actual.level == expected.level
        assert actual.time == expected.time
        assert actual.precursor_mz == expected.precursor_mz
        assert actual.isolation_window == expected.isolation_window
        assert actual.collision_energy == expected.collision_energy
        np.testing.assert_array_equal(actual.signals, expected.signals)


def test_mcraw_random_access_store_is_mmap_backed(tmp_path):
    path = tmp_path / "sample.mcraw"
    save_mcraw(_make_mcraw_data(), path)

    store = McrawStore(path)
    assert len(store) == 4
    assert isinstance(store.signals, np.memmap)
    np.testing.assert_array_equal(store[2], [[100.0, 10.0]])


def test_preprocessed_mcraw_releases_full_backing_arrays(tmp_path):
    path = tmp_path / "sample.mcraw"
    data = _make_mcraw_data()
    save_mcraw(data, path)
    params = Params()
    params.ms1_abs_int_tol = 0
    params.ms2_abs_int_tol = 0
    params.centroid_mz_tol = None

    restored = load_mcraw(path, params=params, preprocess=True, mmap=True)

    assert restored._mcraw_backing_arrays == {}
    assert restored._mcraw_mmap is False
    assert all(not isinstance(scan.signals, np.memmap) for scan in restored.scans)


def test_mcraw_source_fingerprint_detects_stale_cache(tmp_path):
    source = tmp_path / "sample.mzML"
    source.write_bytes(b"first version")
    data = _make_mcraw_data()
    data.update_metadata({"file_path": source})
    cache = tmp_path / "sample.mcraw"
    save_mcraw(data, cache)

    assert mcraw_matches_source(cache, source)
    source.write_bytes(b"a different source version")
    assert not mcraw_matches_source(cache, source)


def test_rt_correction_updates_eic_search_axis():
    data = _make_mcraw_data()
    from masscube.raw_data_utils.core import finalize_scan_indexes

    finalize_scan_indexes(data)
    data.correct_retention_time(lambda value: value + 1.0)

    np.testing.assert_allclose(data.ms1_time_arr, [1.95, 2.00])
    times, signals, _ = data.get_eic_data(100.0, 2.0, mz_tol=0.01, rt_tol=0.1)
    np.testing.assert_allclose(times, [1.95, 2.00])
    np.testing.assert_allclose(signals[:, 1], [5.0, 10.0])


def _feature_with_peak_shape():
    return SimpleNamespace(
        feature_group_id=None,
        id=1,
        mz=100.0,
        rt=1.0,
        adduct_type=None,
        is_isotope=False,
        is_in_source_fragment=False,
        scan_idx=2,
        peak_area=10.0,
        peak_height=5.0,
        top_average=4.0,
        gaussian_similarity=0.9,
        noise_score=0.1,
        asymmetry_factor=1.0,
        charge_state=1,
        isotope_signals=None,
        rt_seq=np.array([0.9, 1.0, 1.1]),
        length=3,
        peak_shape=np.array([[0.9, 1.0], [1.0, 5.0], [1.1, 2.0]]),
        ms2=None,
        matched_ms2=None,
        search_mode=None,
        annotation=None,
        formula=None,
        similarity=None,
        matched_precursor_mz=None,
        matched_peak_number=None,
        smiles=None,
        inchikey=None,
    )


def test_peak_shape_payload_is_opt_in_but_column_is_compatible(tmp_path):
    data = _make_mcraw_data()
    data.features = [_feature_with_peak_shape()]

    compact_path = tmp_path / "compact.txt"
    data.output_single_file(compact_path)
    compact = pd.read_csv(compact_path, sep="\t")
    assert "peak_shape" in compact.columns
    assert pd.isna(compact.loc[0, "peak_shape"])

    full_path = tmp_path / "full.txt"
    data.output_single_file(full_path, include_peak_shape=True)
    full = pd.read_csv(full_path, sep="\t")
    assert full.loc[0, "peak_shape"] == "0.9;1.0|1.0;5.0|1.1;2.0|"
