import os
from pathlib import Path
import pickle

import numpy as np
import pytest

from masscube.params import Params
from masscube.raw_data_utils import (
    MSData,
    Scan,
    TDFReader,
    get_raw_data_format,
    read_raw_file_to_obj,
)


class _UnitFloat(float):
    unit_info = "second"


def test_raw_data_utils_remains_backward_compatible_and_mzml_is_separate(tmp_path):
    data = MSData()
    data.update_metadata({"file_path": tmp_path / "sample.mzML"})
    data.params = Params()
    data.params.scan_levels = [1]
    data.params.ms1_abs_int_tol = 0
    data.params.centroid_mz_tol = None
    spectra = [
        {
            "scanList": {"scan": [{"scan start time": _UnitFloat(90)}]},
            "ms level": 1,
            "m/z array": np.array([100.0, 200.0]),
            "intensity array": np.array([10.0, 20.0]),
        },
        {
            "scanList": {"scan": [{"scan start time": _UnitFloat(91)}]},
            "ms level": 2,
        },
    ]

    data.extract_scans_mzml(spectra)

    assert len(data.scans) == 1
    assert data.scans[0].time == 1.5
    assert data.ms1_idx_arr.tolist() == [0]
    assert data.ms2_idx_arr.tolist() == []
    np.testing.assert_allclose(data.base_peak_arr, [[200.0, 20.0]])


def test_scan_has_stable_optional_mobility_fields():
    scan = Scan()
    assert scan.inv_mobility is None
    assert scan.inv_mobility_range is None
    assert scan.mobility_pressure_compensated is None


def test_pre_refactor_raw_data_utils_pickles_still_load():
    scan = Scan(raw_file_id=7, level=1, scan_time=1.0, signals=np.empty((0, 2)))
    original_module = Scan.__module__
    try:
        Scan.__module__ = "masscube.raw_data_utils"
        payload = pickle.dumps(scan)
    finally:
        Scan.__module__ = original_module

    restored = pickle.loads(payload)
    assert isinstance(restored, Scan)
    assert restored.raw_file_id == 7


def test_d_validation_requires_both_tdf_files(tmp_path):
    data_dir = tmp_path / "sample.d"
    data_dir.mkdir()
    (data_dir / "analysis.tdf").touch()

    with pytest.raises(FileNotFoundError, match="analysis.tdf_bin"):
        get_raw_data_format(data_dir)


_TIMS_TEST_VALUE = os.environ.get("MASSCUBE_TIMS_TEST_DATA")
_TIMS_TEST_DATA = Path(_TIMS_TEST_VALUE) if _TIMS_TEST_VALUE else None


@pytest.mark.skipif(
    _TIMS_TEST_DATA is None or not _TIMS_TEST_DATA.is_dir(),
    reason="set MASSCUBE_TIMS_TEST_DATA to the example Bruker .d directory",
)
class TestBrukerTDFRegression:
    def test_exact_binary_and_physical_calibration(self):
        with TDFReader(_TIMS_TEST_DATA) as reader:
            assert len(reader.frames) == 10_125
            assert reader.total_peaks == 17_847_300
            raw = reader.read_raw_frame(479)
            assert raw.scans[:5].tolist() == [48, 58, 62, 65, 66]
            assert raw.tof_indices[:5].tolist() == [
                361019,
                595689,
                244561,
                168607,
                411528,
            ]
            assert raw.intensities[:5].tolist() == [21, 22, 24, 27, 23]

            calibrated = reader.read_frame(1)
            # Cross-checked against tims_index_to_mz in TDF-SDK 5.0.4.
            assert calibrated.mz_values[0] == pytest.approx(419.6721952788, abs=1e-8)
            assert calibrated.inv_mobility_values[0] == pytest.approx(
                1.4497283794, abs=1e-8
            )
            assert calibrated.pressure_compensated is False

    def test_masscube_mapping_keeps_mobility_aligned(self):
        params = Params()
        params.ms1_abs_int_tol = 0
        params.centroid_mz_tol = None
        params.rt_lower_limit = 74.50 / 60.0
        params.rt_upper_limit = 74.65 / 60.0
        params.scan_levels = [1]

        data = read_raw_file_to_obj(_TIMS_TEST_DATA, params=params)

        assert len(data.scans) == 1
        scan = data.scans[0]
        assert scan.bruker_frame_id == 479
        assert scan.signals.shape == (11_200, 2)
        assert scan.inv_mobility.shape == (11_200,)
        assert scan.inv_mobility_range.shape == (11_200, 2)
        assert np.all(np.diff(scan.signals[:, 0]) >= 0)
        assert data.metadata.mobility_pressure_compensated is False
