# MassCube

[![PyPI Downloads](https://img.shields.io/pypi/dm/bago.svg?label=PyPI%20downloads)](https://pypi.org/project/masscube/)

**masscube** is an integrated Python package for liquid chromatography-mass spectrometry (LC-MS) data processing.

- **Documentation:** https://huaxuyu.github.io/masscubedocs/
- **Source code:** https://github.com/huaxuyu/masscube/
- **Bug reports:** https://github.com/huaxuyu/masscube/issues/

It provides:

- Highly accurate nontargeted peak detection and segmentation.
- Comprehensive feature quality evaluation.
- Confident annotation of feature groups including isotopes, adducts and in-source fragments.
- Annotation of MS/MS spectra via identity search and fuzzy search (i.e. analog search).

## Installation

To install **masscube**, open a terminal and run:

```bash
pip install masscube
```

To upgrade **masscube** to the latest version, run:

```bash
pip install masscube --upgrade
```

## Quick start

Start your nontargeted metabolomics data processing from [here](https://huaxuyu.github.io/masscubedocs/docs/quickstart/)

### Bruker timsTOF `.d` input

MassCube can read DDA-PASEF TDF2 directories directly, including calibrated
m/z and inverse reduced ion mobility (`1/K0`):

```python
from masscube import read_raw_file_to_obj

data = read_raw_file_to_obj("sample.d")
scan = data.scans[data.ms1_idx_arr[0]]
print(scan.signals)             # columns: m/z, intensity
print(scan.inv_mobility)        # aligned intensity-weighted 1/K0
print(scan.inv_mobility_range)  # aligned observed [min, max] 1/K0
```

The reader does not use OpenTIMS, AlphaTims, or Bruker's TDF-SDK. Its binary
decoder uses the Python standard library and loads the system `libzstd`
runtime through `ctypes`; on macOS it can be installed with `brew install
zstd`. Set `MASSCUBE_ZSTD_LIBRARY` when the library is in a nonstandard
location.

Currently supported Bruker data are TDF2 (`TimsCompressionType=2`), DDA-PASEF
(`MsMsType=8`), m/z calibration model 1 with `dC2=0` and `C3=C4=0`, and static
TIMS calibration model 2. The complete model-1 quadratic (`C0`, `C1`, and
`C2`) m/z calibration stored in each frame is applied.
The reported `1/K0` applies the stored static mobility calibration but not the
vendor's proprietary per-frame pressure compensation; this is recorded as
`data.metadata.mobility_pressure_compensated == False`.

For development-time verification against an installed official TDF-SDK, run:

```bash
python tools/validate_tdf_sdk.py sample.d /path/to/libtimsdata.so \
  --output tdf_sdk_validation.json
```

The validation tool compares exact scan/TOF/intensity coordinates before
mobility collapse, calibrated m/z, and all three SDK pressure-compensation
strategies. It is not imported by MassCube and does not make TDF-SDK a runtime
dependency.

### Native `.mcraw` workflow cache

The untargeted workflow decodes each mzML or supported Bruker `.d` source once
and stores a versioned `.mcraw` cache under the project `tmp` directory.
Feature detection, gap filling, and aligned feature grouping then reload this
cache instead of reparsing the source. Shared MS1/MS2 metadata are stored once;
only scan-varying values and concatenated peak arrays are written as NumPy
columns. The original vendor data should still be retained as the archival
source.

Single-file TXT output keeps the `peak_shape` column for compatibility but
does not embed full chromatographic traces by default. Set
`Params.output_peak_shape = True` when those traces are explicitly needed.

## Contribute to masscube

The **masscube** project is excited to have your expertise and passion on board!

We value all enhancements or corrections. For those thinking about making significant contributions to the codebase, we encourage you to get in touch with us!

- Huaxu Yu, huaxuyu@zju.edu.cn; yhxchem@outlook.com

## License

MassCube is licensed under the [Apache License 2.0](LICENSE).
