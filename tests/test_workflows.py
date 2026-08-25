import os
import pickle
import sys
import types

import pandas as pd
import pytest

from masscube.params import Params, PROJECT_CONTEXT_FIELDS
from masscube import project as project_module
from masscube.project import ProjectContext, ProjectMetadata
from masscube import workflows
from masscube.workflows import (
    UntargetedMetabolomicsWorkflow,
    untargeted_metabolomics_workflow,
)


def _prepare_fake_project(tmp_path, with_sample_table=True):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "sample.mzML").touch()
    if with_sample_table:
        pd.DataFrame(
            {
                "sample_name": ["sample"],
                "is_qc": ["yes"],
                "is_blank": ["no"],
                "group": ["A"],
            }
        ).to_csv(tmp_path / "sample_table.csv", index=False)


def _patch_raw_metadata(monkeypatch):
    monkeypatch.setattr(workflows, "DEPENDENCIES", ())
    monkeypatch.setattr(
        workflows,
        "is_supported_raw_data_path",
        lambda path: str(path).endswith(".mzML"),
    )
    monkeypatch.setattr(
        workflows,
        "find_raw_data_info",
        lambda path: ("qtof", "positive", True, 123.0),
    )
    monkeypatch.setattr(workflows, "get_raw_data_format", lambda path: "mzml")
    monkeypatch.setattr(
        project_module,
        "find_raw_data_info",
        lambda path: ("qtof", "positive", True, 123.0),
    )
    monkeypatch.setattr(
        project_module, "get_raw_data_format", lambda path: "mzml"
    )


def test_params_only_contain_processing_parameters():
    params = Params()

    assert not (set(params.__dict__) & PROJECT_CONTEXT_FIELDS)
    assert not (set(params.as_dict()) & PROJECT_CONTEXT_FIELDS)


def test_project_context_owns_project_paths(tmp_path):
    context = ProjectContext(tmp_path)

    assert context.project_dir == str(tmp_path)
    assert context.sample_dir == os.path.join(str(tmp_path), "data")
    assert context.tmp_file_dir == os.path.join(str(tmp_path), "tmp")


def test_workflow_prepare_separates_params_context_and_metadata(tmp_path, monkeypatch):
    _prepare_fake_project(tmp_path)
    _patch_raw_metadata(monkeypatch)

    workflow = UntargetedMetabolomicsWorkflow(tmp_path).prepare()

    assert workflow.context.project_dir == str(tmp_path)
    assert workflow.metadata.project_name == tmp_path.name
    assert workflow.metadata.ion_mode == "positive"
    assert workflow.metadata.sample_names == ["sample"]
    assert workflow.metadata.sample_metadata.loc[0, "is_qc"] == True
    assert workflow.metadata.sample_metadata.loc[0, "ABSOLUTE_PATH"].endswith(
        "sample.mzML"
    )
    assert not (set(workflow.params.__dict__) & PROJECT_CONTEXT_FIELDS)


def test_project_metadata_uses_named_sample_column(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "sample.mzML").touch()
    pd.DataFrame(
        {
            "group": ["A"],
            "Sample_Name": ["sample"],
            "IS_QC": ["yes"],
        }
    ).to_csv(tmp_path / "sample_table.csv", index=False)
    _patch_raw_metadata(monkeypatch)

    metadata = ProjectMetadata.from_sources(
        ProjectContext(tmp_path), ["sample.mzML"], tmp_path / "sample_table.csv"
    )

    assert metadata.sample_names == ["sample"]
    assert metadata.source_paths == [str(data_dir / "sample.mzML")]
    assert metadata.sample_metadata.columns[0] == "sample_name"
    assert metadata.sample_metadata.loc[0, "is_qc"] == True


def test_project_metadata_rejects_missing_or_duplicate_sample_names(
    tmp_path, monkeypatch
):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "sample.mzML").touch()
    context = ProjectContext(tmp_path)
    _patch_raw_metadata(monkeypatch)

    missing_column_path = tmp_path / "missing.csv"
    pd.DataFrame({"file": ["sample"]}).to_csv(missing_column_path, index=False)
    with pytest.raises(ValueError, match="sample_name"):
        ProjectMetadata.from_sources(
            context, ["sample.mzML"], missing_column_path
        )

    duplicate_path = tmp_path / "duplicate.csv"
    pd.DataFrame({"sample_name": ["sample", "sample"]}).to_csv(
        duplicate_path, index=False
    )
    with pytest.raises(ValueError, match="duplicate sample_name"):
        ProjectMetadata.from_sources(context, ["sample.mzML"], duplicate_path)


def test_project_metadata_marks_non_centroid_sources_invalid(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "centroid.mzML").touch()
    (data_dir / "profile.mzML").touch()
    pd.DataFrame(
        {"sample_name": ["centroid", "profile"]}
    ).to_csv(tmp_path / "sample_table.csv", index=False)
    _patch_raw_metadata(monkeypatch)
    monkeypatch.setattr(
        project_module,
        "find_raw_data_info",
        lambda path: (
            "qtof",
            "positive",
            not str(path).endswith("profile.mzML"),
            123.0,
        ),
    )

    metadata = ProjectMetadata.from_sources(
        ProjectContext(tmp_path),
        ["centroid.mzML", "profile.mzML"],
        tmp_path / "sample_table.csv",
    )

    profile = metadata.sample_metadata.set_index("sample_name").loc["profile"]
    assert profile["VALID"] == False
    assert "profile-mode" in profile["INVALID_REASON"]
    assert metadata.source_paths == [str(data_dir / "centroid.mzML")]


def test_project_single_file_processing_writes_status_csv(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source_path = data_dir / "sample.mzML"
    source_path.touch()

    workflow = UntargetedMetabolomicsWorkflow(tmp_path)
    workflow.context.create_output_directories()
    workflow.metadata = ProjectMetadata(
        project_name=tmp_path.name,
        sample_metadata=pd.DataFrame(
            {
                "sample_name": ["sample"],
                "VALID": [True],
                "INVALID_REASON": [None],
                "ABSOLUTE_PATH": [str(source_path)],
                "MS_TYPE": ["qtof"],
                "ION_MODE": ["positive"],
                "IS_CENTROID": [True],
                "FILE_FORMAT": ["mzml"],
                "time": [None],
            }
        ),
    )
    workflow._prepared = True
    workflow.state.run_id = "run-1"
    workflow.params.percent_cpu_to_use = 1.0

    class ImmediateParallel:
        def __init__(self, **_kwargs):
            pass

        def __call__(self, tasks):
            return (
                function(*args, **kwargs)
                for function, args, kwargs in tasks
            )

    monkeypatch.setattr(workflows, "Parallel", ImmediateParallel)
    monkeypatch.setattr(workflows, "_new_run_id", lambda: "run-1")
    monkeypatch.setattr(workflows.multiprocessing, "cpu_count", lambda: 1)
    monkeypatch.setattr(
        workflows, "_mcraw_cache_is_current", lambda *_args: False
    )
    monkeypatch.setattr(
        workflows,
        "_prepare_project_mcraw_cache",
        lambda source, *_args: {
            "sample_name": "sample",
            "cache_path": str(tmp_path / "tmp" / "sample.mcraw"),
            "reason": "",
        },
    )
    monkeypatch.setattr(
        workflows,
        "_process_project_single_file",
        lambda *_args, **_kwargs: {
            "sample_name": "sample",
            "status": "completed",
            "reason": "",
            "feature_count": 3,
            "started_at": "start",
            "ended_at": "end",
        },
    )

    workflow.process_single_files()

    status_path = tmp_path / "project_files" / "single_file_processing_status.csv"
    status = pd.read_csv(status_path)
    assert list(status.columns) == list(workflows.SINGLE_FILE_STATUS_COLUMNS)
    assert status.loc[0, "status"] == "completed"
    assert status.loc[0, "feature_count"] == 3
    assert status.loc[0, "run_id"] == "run-1"
    assert workflow.state.artifacts["single_file_processing_status"] == str(
        status_path
    )
    assert workflow.state.processing_metadata[2]["file_status_counts"] == {
        "completed": 1
    }
    assert not list(status_path.parent.glob("*.tmp-*"))


def test_existing_single_file_status_controls_reuse(tmp_path):
    single_file_dir = tmp_path / "single_files"
    single_file_dir.mkdir()
    result_path = single_file_dir / "sample.txt"
    result_path.write_text("feature_ID\n1\n2\n", encoding="utf-8")
    sample_metadata = pd.DataFrame(
        {
            "sample_name": ["sample"],
            "VALID": [True],
            "INVALID_REASON": [None],
            "ABSOLUTE_PATH": [str(tmp_path / "sample.mzML")],
        }
    )
    previous = pd.DataFrame(
        [
            {
                "sample_name": "sample",
                "source_path": str(tmp_path / "sample.mzML"),
                "result_path": str(result_path),
                "status": "skipped",
                "reason": "no selected MS1 scans",
                "feature_count": None,
                "started_at": "",
                "ended_at": "",
                "run_id": "old-run",
            }
        ]
    )

    reusable = workflows._reusable_single_file_results({"sample"}, previous)
    status = workflows._initialize_single_file_status(
        sample_metadata=sample_metadata,
        single_file_dir=single_file_dir,
        cache_was_current={"sample": True},
        reusable_files=reusable,
        previous_status=previous,
        run_id="new-run",
    )

    assert reusable == set()
    assert status.loc[0, "status"] == "pending"

    reusable = workflows._reusable_single_file_results({"sample"}, None)
    status = workflows._initialize_single_file_status(
        sample_metadata=sample_metadata,
        single_file_dir=single_file_dir,
        cache_was_current={"sample": True},
        reusable_files=reusable,
        previous_status=None,
        run_id="new-run",
    )
    assert status.loc[0, "status"] == "reused"
    assert status.loc[0, "feature_count"] == 2


def test_resumed_workflow_reloads_features_and_persists_object(tmp_path, monkeypatch):
    _prepare_fake_project(tmp_path)
    _patch_raw_metadata(monkeypatch)
    project_file_dir = tmp_path / "project_files"
    project_file_dir.mkdir()
    pd.DataFrame({"feature_ID": [1]}).to_csv(
        tmp_path / "aligned_feature_table.txt", sep="\t", index=False
    )
    expected_features = [{"id": 1}]
    with open(project_file_dir / "aligned_features.pkl", "wb") as file:
        pickle.dump(expected_features, file)

    monkeypatch.setattr(
        UntargetedMetabolomicsWorkflow,
        "process_single_files",
        lambda self: self,
    )
    features, params = untargeted_metabolomics_workflow(
        str(tmp_path), return_results=True
    )

    assert features == expected_features
    assert isinstance(params, Params)
    saved = UntargetedMetabolomicsWorkflow.load(tmp_path)
    assert saved.features == expected_features
    assert saved.metadata.sample_names == ["sample"]
    assert saved.state.status == "completed"


def test_functional_return_params_only_keeps_params_pure(tmp_path, monkeypatch):
    _prepare_fake_project(tmp_path, with_sample_table=False)
    _patch_raw_metadata(monkeypatch)

    params = untargeted_metabolomics_workflow(tmp_path, return_params_only=True)

    assert isinstance(params, Params)
    assert not (set(params.__dict__) & PROJECT_CONTEXT_FIELDS)


def test_load_migrates_project_workflow_module_path(tmp_path):
    project_file_dir = tmp_path / "project_files"
    project_file_dir.mkdir()
    project_path = project_file_dir / "project.masscube"
    workflow = UntargetedMetabolomicsWorkflow(tmp_path)

    old_module_name = "masscube.project_workflow"
    original_module_name = UntargetedMetabolomicsWorkflow.__module__
    legacy_module = types.ModuleType(old_module_name)
    legacy_module.UntargetedMetabolomicsWorkflow = UntargetedMetabolomicsWorkflow
    sys.modules[old_module_name] = legacy_module
    UntargetedMetabolomicsWorkflow.__module__ = old_module_name
    try:
        with open(project_path, "wb") as file:
            pickle.dump(workflow, file)
    finally:
        UntargetedMetabolomicsWorkflow.__module__ = original_module_name
        sys.modules.pop(old_module_name, None)

    loaded = UntargetedMetabolomicsWorkflow.load(project_path)

    assert isinstance(loaded, UntargetedMetabolomicsWorkflow)
    assert loaded.project_dir == str(tmp_path)
