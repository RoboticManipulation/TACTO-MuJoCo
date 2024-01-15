# TACTO: A Fast, Flexible and Open-source Simulator for High-Resolution Vision-based Tactile Sensors

[![License: MIT](https://img.shields.io/github/license/facebookresearch/tacto)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/tacto)](https://pypi.org/project/tacto/)
[![CircleCI](https://circleci.com/gh/facebookresearch/tacto.svg?style=shield)](https://circleci.com/gh/facebookresearch/tacto)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<a href="https://digit.ml/">
<img height="20" src="/website/static/img/digit-logo.svg" alt="DIGIT-logo" />
</a>


This package is a forked version of the repository [tacto](https://github.com/facebookresearch/tacto). It provides a simulator for vision-based tactile sensors, such as [DIGIT](https://digit.ml) with Mujoco.


For more information refer to the corresponding paper [TACTO: A Fast, Flexible, and Open-source Simulator for High-resolution Vision-based Tactile Sensors](https://arxiv.org/abs/2012.08456).

NOTE: the simulator is not meant to provide a physically accurate dynamics of the contacts (e.g., deformation, friction), but rather relies on existing physics engines.

## Installation
1. ``Fresh environment with python 3.8``
2. ``pip install -r requirements.txt``
3. ``pip install dm-control==1.0.14``
   + Ignore warnings related to pyopengl
4. ``pip install pyopengl==3.1.4``
   + Ignore warnings related to pyopengl
5. Install OSMesa

## Usage
Run demo by executing ``python demo_mujoco_digit.py``

## License
This project is licensed under MIT license, as found in the [LICENSE](LICENSE) file.

## Citing
If you use this project in your research, please cite:

```BibTeX
@Article{Wang2022TACTO,
  author   = {Wang, Shaoxiong and Lambeta, Mike and Chou, Po-Wei and Calandra, Roberto},
  title    = {{TACTO}: A Fast, Flexible, and Open-source Simulator for High-resolution Vision-based Tactile Sensors},
  journal  = {IEEE Robotics and Automation Letters (RA-L)},
  year     = {2022},
  volume   = {7},
  number   = {2},
  pages    = {3930--3937},
  issn     = {2377-3766},
  doi      = {10.1109/LRA.2022.3146945},
  url      = {https://arxiv.org/abs/2012.08456},
}
```

