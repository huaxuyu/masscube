import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from masscube.alignment import gap_filling
from masscube.mzpkl import convert_MSData_to_mzpkl, read_mzpkl_to_MSData
from masscube.raw_data_utils import MSData, Scan, save_mcraw


def _make_msdata():
    data = MSData()
    data.update_metadata({"file_path": "/raw/sample.mzML", "ion_mode": "positive"})
    data.scans = [
        Scan(
            raw_file_id=index,
            level=1,
            scan_time=scan_time,
            signals=np.array([[100.0, intensity]], dtype=np.float32),
        )
        for index, (scan_time, intensity) in enumerate(
            [(0.95, 5.0), (1.00, 10.0), (1.05, 7.0)]
        )
    ]
    data.ms1_idx_arr = np.array([0, 1, 2], dtype=np.int32)
    data.ms2_idx_arr = np.array([], dtype=np.int32)
    data.ms1_time_arr = np.array([0.95, 1.00, 1.05], dtype=np.float32)
    return data


def test_mcraw_round_trip_supports_gap_filling(tmp_path):
    mcraw_path = tmp_path / "sample.mcraw"
    data = _make_msdata()
    from masscube.params import Params

    data.params = Params()
    save_mcraw(data, mcraw_path)

    from masscube.raw_data_utils import load_mcraw

    restored = load_mcraw(mcraw_path)
    assert restored.metadata.file_name == "sample"
    times, signals, _ = restored.get_eic_data(100.0, 1.0, mz_tol=0.01, rt_tol=0.1)
    np.testing.assert_allclose(times, [0.95, 1.00, 1.05])
    np.testing.assert_allclose(signals[:, 1], [5.0, 10.0, 7.0])

    feature = SimpleNamespace(
        mz=100.0,
        rt=1.0,
        feature_id_arr=np.array([-1], dtype=int),
        peak_height_arr=np.array([0.0]),
        peak_area_arr=np.array([0.0]),
        top_average_arr=np.array([0.0]),
    )
    params = SimpleNamespace(
        gap_filling_method="local_maximum",
        correct_rt=False,
        project_file_dir=str(tmp_path),
        tmp_file_dir=str(tmp_path),
        sample_metadata=pd.DataFrame({"sample_name": ["sample"], "is_blank": [False]}),
        mz_tol_alignment=0.01,
        gap_filling_rt_window=0.1,
    )

    gap_filling([feature], params)

    assert feature.peak_height_arr[0] == 10.0
    assert feature.detection_rate_gap_filled == 1.0


def test_mzpkl_loader_migrates_legacy_scan_id(tmp_path):
    data = _make_msdata()
    legacy_scan = data.scans[0]
    legacy_scan.id = legacy_scan.raw_file_id
    del legacy_scan.raw_file_id
    mzpkl_path = tmp_path / "legacy.mzpkl"
    convert_MSData_to_mzpkl(data, mzpkl_path)

    restored = read_mzpkl_to_MSData(MSData(), mzpkl_path)

    assert restored.scans[0].raw_file_id == 0


def test_gap_filling_applies_saved_rt_model_to_mcraw_search_axis(tmp_path):
    data = _make_msdata()
    from masscube.params import Params

    data.params = Params()
    save_mcraw(data, tmp_path / "sample.mcraw")
    model = interp1d(
        [0.0, 2.0],
        [1.0, 3.0],
        bounds_error=False,
        fill_value="extrapolate",
    )
    with open(tmp_path / "rt_correction_models.pkl", "wb") as stream:
        pickle.dump({"sample": model}, stream)

    feature = SimpleNamespace(
        mz=100.0,
        rt=2.0,
        feature_id_arr=np.array([-1], dtype=int),
        peak_height_arr=np.array([0.0]),
        peak_area_arr=np.array([0.0]),
        top_average_arr=np.array([0.0]),
    )
    params = SimpleNamespace(
        gap_filling_method="local_maximum",
        correct_rt=True,
        project_file_dir=str(tmp_path),
        tmp_file_dir=str(tmp_path),
        sample_metadata=pd.DataFrame(
            {"sample_name": ["sample"], "is_blank": [False]}
        ),
        mz_tol_alignment=0.01,
        gap_filling_rt_window=0.1,
    )

    gap_filling([feature], params)

    assert feature.peak_height_arr[0] == 10.0
