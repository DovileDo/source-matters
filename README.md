# Source Matters: Source Dataset Impact on Model Robustness in Medical Imaging

This repository contains the code and results included in the paper (currently under review, link coming later).

Provided are the implementations of:

* Confounder generation,
* fine-tuning, and
* performance analysis as described in the paper.

For access to the data used in the paper, please refer to:
* [RadImageNet](https://github.com/BMEII-AI/RadImageNet) for pre-trained weights,
* [NIH CXR14](https://nihcc.app.box.com/v/ChestXray-NIHCC), and
* [LIDC-IDRI](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254).

Our dataset splits are provided in the data folder.

The main files to run are:

`include_artifacts.py`, `include_lowpass.py`, `include_noise.py`, `gender.py` (for confounder generation as described in the paper)
`fine-tuning.py` (for fine-tuning the models on confounded targets and logging results)
`analysis.py` (for pulling results together and plotting)

Feel free to contact us for help with the reproduction of our experiments.
