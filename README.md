# MetabEngine

[![Generic badge](https://img.shields.io/badge/metabengine-ver_0.0.16-%3CCOLOR%3E.svg)](https://github.com/Waddlessss/metabengine/)
![Maintainer](https://img.shields.io/badge/maintainer-Huaxu_Yu-blue)
[![PyPI Downloads](https://img.shields.io/pypi/dm/bago.svg?label=PyPI%20downloads)](https://pypi.org/project/metabengine/)

**metabengine** is an integrated Python package for liquid chromatography-mass spectrometry (LC-MS) data processing.

* **Documentation:** https://metabengine.readthedocs.io/en/latest/
* **Source code:** https://github.com/Waddlessss/metabengine/
* **Bug reports:** https://github.com/Waddlessss/metabengine/issues/

It provides:

* Ion-identity-informed chromatograpgic peak picking.
* Peak quality evaluation via artificial neural network.
* Accurate annotation of isotopes, adducts, and in-source fragments.
* Ultra-fast annotation of MS/MS spectra supported by [Flash Entropy Search](https://github.com/YuanyueLi/MSEntropy)

## Installation

```sh
# PyPI
pip install metabengine
```

The changes to **metabengine** between each release can be found [here](https://pypi.org/project/metabengine/#history). See more from the commit logs.

## Quick start

Over 100 fundemental functions and objects are available in **metabengine** to help you create the best data processing workflow for your study. Some examples of the pre-made workflows are here:

* Untargeted metabolomics workflow
* Data quality examination
* MS/MS search (identity search and hybrid search)
* Visualize MS data and generate plots ready for publication

## Contribute to metabengine

The **metabengine** project is excited to have your expertise and passion on board!

We value all enhancements or corrections. For those thinking about making significant contributions to the codebase, we encourage you to get in touch with us!

* Huaxu Yu, hxuyu@ucdavis.edu