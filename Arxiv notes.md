# Arxiv 论文记录

## A Wireless-Vision Dataset for Privacy Preserving Human Activity Recognition

Propose a dataset for privacy-preserving using a WIFI signal. [link](https://arxiv.org/abs/2205.11962)

## SCVRL: Shuffled Contrastive Video Representation Learning

Based on CVRL, to boost motion patterns, when capturing negative samples, use the same video with shuffled frames. Called 'shuffled contrastive learning'. The negative samples consist of shuffled clips from same video and normal clips from different videos. [link](https://arxiv.org/abs/2205.11710)

## mPLUG: Effective and Efficient Vision-Language Learning by Cross-modal Skip-connections

Large-scale cross-modal pretraining. A unimodal encoder for each modality and cross-attention to fuse latent. Use skip connection for a better modality fuse. [link](https://arxiv.org/abs/2205.12005)

## Learning Muti-experts Distribution Calibration for Long-tailed Video Classification

Multi-experts calibration modeling intra-class and inter-class distribution. [link](https://arxiv.org/abs/2205.10788)

## Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention

Transform standard self-attention to linear complicity by matrix decomposition. [link](https://arxiv.org/abs/2102.03902)

A similar approach is Performer. [link](https://arxiv.org/abs/2009.14794)

## Inception Transformer

Propose iFormer, adopts a channel splitting mechanism to couple convolution/ pooling and self-attention, giving more concentrations on high-freq and expanding the perception capability of the Transformer in the frequency spectrum. [link](https://arxiv.org/abs/2205.12956)

## Contrastive Learning with Boosted Memorization

To tackle the long-tailed distribution in real-world problems. Proposing a Boosted Contrastive Learning method that automatically drives the information discrepancy(信息差异) of sample views in contrastive learning. Trace the historical losses of each sample to find the clues about the memorization effect of the model and adaptively control the augmentation strength to enhance the learning law on the tail samples. [link](https://arxiv.org/abs/2205.12693)

## Cross-Architecture Self-supervised Video Representation Learning

Cross architecture contrastive learning, consensus 3DCNN and transformer to generate diverse yet meaningful positive pairs. Also, propose a novel temporal learning method measuring an edit distance between a video and its temporal self-shuffle.

Kinda like 3D CNN & transformer co-training. [link](https://arxiv.org/abs/2205.13313)

## AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition

Adapting a ViT to various image and video tasks is challenging due to the large cost of computation & storage burdens. Especially for each model has to be independently fine-tuned to different tasks, limiting its transferability in different domains. Proposing AdaptFormer architecture that can adapt pre-trained ViTs by fine-tuning less than 2% parameters in downstream tasks. [link](https://arxiv.org/abs/2205.13535)

## SupMAE: Supervised Masked Autoencoders Are Efficient Vision Learners

Use supervised signal and classification loss to guide the encoder's pre-training. The overall object is classification & reconstruction loss. [link](https://arxiv.org/pdf/2205.14540.pdf)

## IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation

Video frame interpolation. Maybe check later. [link](https://arxiv.org/abs/2205.14620)

## From Representation to Reasoning: Towards both Evidence and Commonsense Reasoning for Video Question-Answering

Propose a novel benchmark for Video Q&A tasks. [link](https://arxiv.org/abs/2205.14895)

## Multimodal Masked Autoencoders Learn Transferable Representations

Text & image MAE. Using a crossmodal encoder to encode the presentation of text and image. [link](https://arxiv.org/pdf/2205.14204.pdf)

## Self-Supervised Visual Representation Learning with Semantic Grouping

The semantic grouping (feature-space pixel-level deep clustering) is performed by assigning pixels to a set of learnable prototypes, which can adapt to each sample by attentive pooling over the feature and forming new slots. 

## CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers

Text-video generation by VQVAE. [link](https://arxiv.org/pdf/2205.15868.pdf)

## Contrastive Principal Component Learning: Modeling Similarity by Augmentation Overlap

The semantic relationship between samples should be considered during contrastive learning. Employing the idea of PCA on augmentation feature that encodes information about the augmentation distributions. [link](https://arxiv.org/pdf/2206.00471.pdf)

## Siamese Image Modeling for Self-Supervised Vision Representation Learning

(Jifeng Dai)The author thinks that contrasting different augmented views can help to lean semantic alignment. So they propose siamese image modeling, using a masked online branch and a full augmented target branch to calculate Dense loss. [link](https://arxiv.org/pdf/2206.01204.pdf)

## Hard Negative Sampling Strategies for Contrastive Representation Learning

Hard samples: close to the decision boundary and far from each other. Calculating the contrastive loss between hard samples may help contrastive learning. [link](https://arxiv.org/pdf/2206.01197.pdf)

## Revisiting the “Video” in Video-Language Understanding

(Feifei Li)Propose the atemporal probe (ATP) for video-language analysis. Provide a strong baseline for video-language tasks. [link](https://arxiv.org/pdf/2206.01720.pdf)

## Egocentric Video-Language Pretraining

(ShowLab) 1. Create EgoClip, a new dataset for video-text pretraining (a subset of Ego4D). 2. Propose EgoNCE that adapts video-text contrastive learning to egocentric domain by mining egocentric-aware positive and negative samples. [link](https://arxiv.org/pdf/2206.01670.pdf)

## Rethinking Positive Sampling for Contrastive Learning with Kernel

Propose a new way to define positive samples using kernel theory along with a novel loss called decoupled uniformity. [link](https://arxiv.org/pdf/2206.01646.pdf)

## Team VI-I2R Technical Report on EPIC-KITCHENS-100 Unsupervised Domain Adaptation Challenge for Action Recognition 2021

Technical report. Contains hand-centric feature generation & video domain adaptation. For feature generation part, using joint TBN & TSM architecture; for the domain adaptation part, adapt TA3N. [link](https://arxiv.org/pdf/2206.02573.pdf)

## Invariant Grounding for Video Question Answering

In VideoQA tasks, empirical risk minimization may lead to some problems, because it tends to over-exploit the spurious correlations between question irrelevant scenes and answers. Authors partition the visual scenes into two parts:1) the visual scene, and 2) it's component which is irrelevant to the answer. The author claims that the causal scene is expected to be sufficient and necessary to answer the question, and no critical clues should exist in the complement scene to answer the question. [link](https://arxiv.org/pdf/2206.02349.pdf)

*VideoQA common paradigm: 1) video-question joint encoder, encapsulates the visual scenes of video and the linguistic semantics of question as representations. 2) answer decoder, exploits the latent to model the visual-linguistic alignment and yield an answer.* 

## Causal Attention for Unbiased Visual Recognition

In IID settings, a model equipped with attention is better; however, in OOD (out of distribution) settings, the attention model is even worse than the non-attention baseline. Authors propose a visual attention module CaaM that learns causal features that are robust in OOD settings without sacrificing the performance in IID settings.

## Rethinking the Openness of CLIP

Though CLIP shows great potential in realizing open-vocabulary image classification in a matching style (e.g. zero-shot image classification/ video classification /open set classification ...), its performances degrade as the vocabulary expands to different degrees, and that is due to the confusion among competing text features, i.e. not stable with respect to the vocabulary. Enforcing the distinguishability of text features could improve the openness of CLIP-based model (even without fine-tuning). [link](https://arxiv.org/pdf/2206.01986.pdf)

1. Traditional evaluation protocols are not sufficient for the open recognition tasks as it has limited and fixed target vocabulary. So 2 novel evaluation protocols are proposed: extensibility & stability.

2. The small margin between text vocabulary leads to the poor stability of CLIP model. It shows in the confusion matrix that the positive pairs' cosine similarity and negative pairs' cosine similarity are very close.

3. CoOp could improve the inter-modal alignment that text feature is towards the cluster center of the corresponding image features. 

4. The original text prompt like "a photo of a [CLASS]" is not optimal, as the same context in the prompt cannot provide holistic and diverse semantics modeling for different visual categories. A better way is to retrieve the captions from pre-training dataset as a prompt ensemble. (The most similar images)


## On the duality between contrastive and non-contrastive self-supervised learning

