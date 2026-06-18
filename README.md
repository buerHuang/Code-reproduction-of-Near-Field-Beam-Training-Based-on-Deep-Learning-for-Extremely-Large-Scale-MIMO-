# Code reproduction of Near Field Beam Training Based on Deep Learning for Extremely Large Scale MIMO

![DNBTnet](https://github.com/buerHuang/Code-reproduction-of-Near-Field-Beam-Training-Based-on-Deep-Learning-for-Extremely-Large-Scale-MIMO-/blob/main/deep_learning.svg)

**This is an example of Code reproduction of the paper “Near-Field Beam Training Based on Deep Learning for Extremely Large-Scale MIMO”**

Link: https://ieeexplore.ieee.org/document/10163797

Citations: G. Jiang and C. Qi, "Near-field beam training based on deep learning for extremely large-scale MIMO," _IEEE Commun. Lett._, vol. 27, no. 8, pp. 2063-2067, Aug. 2023.

near_field_channel: Generates the near-field channel;

near_field_manifold: Generates near-field array response vectors;

polar_codebook: Creates a near-field polar domain codebook where angles are sampled uniformly and distances are sampled nonuniformly;

polar_domain_manifold: Creates near-field polar-domain array response vectors that match near-field spherical wave channels;

**codeword_sampling: Based on the content of the paper, the codeword sparse sampling of the near-field codebook is performed for subsequent training;**

**dataset_generator: The random channels are generated and the best codeword matched by exhaustive beam training is used as the training index;**

**net: The deep learning model based on the content of the paper;**

training_near_exhaustive: Near-field exhaustive beam training method, as an upper bound for codeword sampling;

train_dnbt: Trains and saves the deep learning model;

Rate_SNR: Example performance simulation code.
