# Semantic-Aware Cropping for Face Foundation Models

This repository contains the code for Team 006's HLCV project. We investigate the impact of semantic-aware cropping on the latent representations of state-of-the-art Face Foundation Models (FaceX-Former, AdaFace).

## Project Structure
* `data_pipeline/`: Dataloaders for CelebAMask-HQ and our 5 custom augmentation/occlusion pipelines.
* `models_and_probing/`: Scripts to freeze backbones, extract multi-scale token embeddings (.pt files), and train lightweight MLPs.
* `analysis/`: Code to compute Cosine Similarity and the Region Preservation Ratio.
* `cluster_configs/`: HTCondor `.sub` and bash scripts for running jobs on the UdS GPU cluster.

## Setup Instructions
1. Clone the repository on the UdS cluster.
2. Build the conda environment: `conda env create -f cluster_configs/environment.yml`
3. Activate the environment: `conda activate hlcv`
4. Submit jobs using: `condor_submit cluster_configs/pytorch_docker.sub`

semantic-face-crop/
├── data_pipeline/
│   ├── celeb_loader.py
│   └── augmentations.py
├── models_and_probing/
│   └── extract_features.py
├── analysis/
│   └── metrics.py
├── cluster_configs/
│   ├── execute.sh
│   ├── pytorch_docker.sub
│   └── environment.yml
└── README.md



