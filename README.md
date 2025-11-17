# STHG-deepfake-detection
## The codes of STHG: Spatial-Temporal Hypergraph for Deepfake Detection (Submission 10789) ##

1. Our codes is based on widely used codebase [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench?tab=readme-ov-file).
Firat of all, plealse follow it's offical guideness for [Installation](https://github.com/SCLBD/DeepfakeBench?tab=readme-ov-file#1-installation), [Download Data](https://github.com/SCLBD/DeepfakeBench?tab=readme-ov-file#2-download-data), [Preprocessing (optional)](https://github.com/SCLBD/DeepfakeBench?tab=readme-ov-file#3-preprocessing-optional) and [Rearrangement](https://github.com/SCLBD/DeepfakeBench?tab=readme-ov-file#4-rearrangement).

2. Training 

>     python training/train.py --detector_path ./training/config/detector/sthhg.yaml

3. Evaluation

>     python training/test.py \
>     --detector_path ./training/config/detector/sthhg.yaml \
>     --test_dataset "Celeb-DF-v2" "DFDC"\
>     --weights_path [path to weights]
