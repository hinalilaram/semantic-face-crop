# Semantic-Aware Cropping for Face Foundation Models

In this repository we investigate the impact of semantic-aware cropping on the latent representations of state-of-the-art Face Foundation Models (FaceX-Former and Arc2Face).

## Project Structure
* `data/`: Dataloaders for CelebAMask-HQ and our 5 custom augmentation/occlusion pipelines.
* `models/`: Scripts to freeze backbones, extract multi-scale token embeddings (.pt files), and train lightweight MLPs.
* `analysis/`: Code to compute Cosine Similarity and the Region Preservation Ratio.
* `cluster/`: HTCondor `.sub` and bash scripts for running jobs on the UdS GPU cluster.

```
semantic-face-crop/
├── data/
│   ├── celeb_loader.py
│   └── augmentations.py
├── models/
│   └── extract_features.py
├── analysis/
│   └── metrics.py
├── cluster/
│   ├── execute.sh
│   ├── pytorch_docker.sub
│   └── environment.yml
└── README.md
```


