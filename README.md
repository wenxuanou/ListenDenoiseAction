# Listen, Denoise, Action!
This repository provides code and models for the paper [Listen, denoise, action! Audio-driven motion synthesis with diffusion models](https://arxiv.org/abs/2211.09707).

Please watch the following video for an introduction to the paper:
* [SIGGRAPH 2023 presentation](https://youtu.be/Qfd2EpzWgok)

For video samples and a general overview, please see [our project page](https://www.speech.kth.se/research/listen-denoise-action/).

## Installation
We provide a Docker file and `requirements.txt` for installation using a Docker image or Conda.

### Installation using Conda
```
conda install python=3.9
conda install -c conda-forge mpi4py mpich
pip install -r requirements.txt
```

## Dance synthesis demo
### Data and pretrained models
Please [download our pretrained dance models here](https://zenodo.org/record/8156769) and move them to the `pretrained_models` folder.
We include processed music inputs from the test dataset in the `data` folder for generating dances from the model.

### Synthesis scripts
You can use the following shell scripts for reproducing the dance user studies in the paper:
```
./experiments/dance_LDA.sh
./experiments/dance_LDA-U.sh
```
To try out locomotion synthesis, please go to https://www.motorica.ai/.

## Training data
The four main training datasets from our SIGGRAPH 2023 paper are available online:
* [The Trinity Speech Gesture Dataset](https://trinityspeechgesture.scss.tcd.ie/)
* [The ZEGGS dataset](https://github.com/ubisoft/ubisoft-laforge-ZeroEGGS)
* [The 100STYLE dataset](https://www.ianxmason.com/100style/)
* [The Motorica Dance Dataset](https://github.com/simonalexanderson/MotoricaDanceDataset/), a new dataset with high-quality dance mocap released together with our paper

## License and copyright information
The contents of this repository may not be used for any purpose other than academic research. It is free to use for research purposes by academic institutes, companies, and individuals. Use for commercial purposes is not permitted without prior written consent from Motorica AB. If you are interested in using the codebase, pretrained models or the dataset for commercial purposes or non-research purposes, please contact us at info@motorica.ai in advance. Unauthorised redistribution is prohibited without written approval.

### Attribution
Please include the following citations in any preprints and publications that use this repository.
```
@article{alexanderson2023listen,
  title={Listen, Denoise, Action! Audio-Driven Motion Synthesis with Diffusion Models},
  author={Alexanderson, Simon and Nagy, Rajmund and Beskow, Jonas and Henter, Gustav Eje},
  year={2023}
  issue_date={August 2023},
  publisher={ACM},
  volume={42},
  number={4},
  doi={10.1145/3592458},
  journal={ACM Trans. Graph.},
  articleno={44},
  numpages={20},
  pages={44:1--44:20}
}
```
The [code for translation-invariant self-attention](https://github.com/ulmewennberg/tisa) (TISA) was written by [Ulme Wennberg](https://www.kth.se/profile/ulme). Please cite [the correspoding ACL 2021 article](https://aclanthology.org/2021.acl-short.18) if you use this code.
