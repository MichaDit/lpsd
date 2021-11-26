# Python 3 LPSD algorithm

## Overview

This repository contains a Python 3 implementation of the LPSD algorithm.
The implementation is similiar to the LPSD implementation in the Matlab package LTPDA.
The core of the algorithm can be run as Python 3 or as (faster) C code.
Both deliver the *same results*.
To run the C core the file ltpda_dft.c has to be compiled to a shared library.

## Installation

Install the `lpsd-*.whl` file with `pip` or use
```bash
pip install lpsd-package-does-not-exist-yet
```
TODO

## Usage

TODO

## References
[Improved spectrum estimation from digitized time series on a logarithmic frequency axis](https://doi.org/10.1016/j.measurement.2005.10.010)
Authors: Michael Tröbs and Gerhard Heinzel

[Spectrum and spectral density estimation by the Discrete Fourier transform (DFT), including a comprehensive list of window functions and some new flat-top windows](http://hdl.handle.net/11858/00-001M-0000-0013-557A-5)
Authors: Gerhard Heinzel, Albrecht Rüdiger and Roland Schilling

[MATLAB Toolbox LTPDA](https://www.elisascience.org/ltpda/)
Author:  Martin Hewitson
