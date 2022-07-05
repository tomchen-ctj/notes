# Daily Arxiv Notes

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

## Can CNNs Be More Robust Than Transformers?

With three simple changes, CNN could be as robust as Transformer architecture. 1) patchifying input images, 2) enlarging kernel size, 3) reducing activation layers and normalization layers. Compared to ConvNeXt, this work mainly focus on the robustness of the model.   [link](https://arxiv.org/pdf/2206.03452.pdf)

## TRIBYOL: TRIPLET BYOL FOR SELF-SUPERVISED REPRESENTATION LEARNING

TriBYOL performs better with small batch sizes. One online network and two target networks ( do not share weights). [link](https://arxiv.org/pdf/2206.03012.pdf)

## Revealing Single Frame Bias for Video-and-Language Learning

Training an effective video-and-language model intuitively requires multiple frames as model inputs. But it is unclear that using multiple frames could benefit downstream tasks. Actually, with large-scale pre-training and a proper frame ensemble strategy at inference time, only using one frame performs even better than multiple frames training. 

In training stage, a frame is randomly chosen from a video clip and used for training. During inferece, a list of frames are uniformly sampled and early fuse is used to fuse their encoded image-level representations as input to the multi-modal encoder.  [link](https://arxiv.org/pdf/2206.03428.pdf)

## Extreme Masking for Learning Instance and Distributed Visual Representations

(Zhirong Wu, MSRA) A scalable approach for learning distributed representations over individual tokens. Follows the architecture of BYOL, the target encoder processes the full view while the base encoder prosses a partial view from extreme masked sampling. In this way, the overall complicity of base encoder is greatly reduced. Also, this could naturally provide different-but-correlational views to boost the representation learning.  [link](https://arxiv.org/pdf/2206.04667.pdf)

## SimVP: Simpler yet Better Video Prediction

(Stan Z.Li, Westlake Univ.) Using a very simple architecture (CNN- CNN - CNN) and no extra tricks get sota results on 5 video prediction benchmarks. [link](https://arxiv.org/pdf/2206.05099.pdf)

## An Empirical Study on Disentanglement of Negative-free Contrastive Learning

This work aims at the disentanglement of negative-free contrastive learning methods. Proposes a new disentanglement metric based on Mutual Information between representation and data factors. 

1. Contrastive learning without negatives can learn a well-disentangled subspace of latent representation. 

2.  Existing low-dimensional generative disentangled learning methods can not learn good representation in real-world datasets.

[link](https://arxiv.org/pdf/2206.04756.pdf)

## Lost in Transmission: On the Impact of Networking Corruptions on Video Machine Learning Models

Networking corruptions could apparently affect video models. The corruptions cause visual and temporal artifacts i.e. smeared colors (mosaic) or frame drops. [link](https://arxiv.org/pdf/2206.05252.pdf)

## Spatial Cross-Attention Improves Self-Supervised Visual Representation Learning

Proposes a novel add-on module to facilitate the injection of the knowledge accounting for spatial correlations among the samples. The output of the module is seen as a mask to guide the feature maps. (Prototypes clustering) [link](https://arxiv.org/pdf/2206.05028.pdf)

## Learn2Augment: Learning to Composite Videos for Data Augmentation in Action Recognition

Using a network to decouple the foreground and the background (for background scene, apply image inpainting) [link](https://arxiv.org/pdf/2206.04790.pdf)

## Is Self-Supervised Learning More Robust Than Supervised Learning?

Contrastive learning is prone to be attacked by patch shuffling and pixel intensity change, yet less sensitive to dataset-level distribution change. Besides, contrastive learning is more robust than supervised learning under downstream corruptions. [link](https://arxiv.org/pdf/2206.05259.pdf)

## TRANSDUCTIVE CLIP WITH CLASS-CONDITIONAL CONTRASTIVE LEARNING

Leveraging the supervision of CLIP is a great way to alleviate the burden of data labeling. But directly utilizing the pseudo label from CLIP model may cause the student model to learn label noises. 

1. A class conditional contrastive learning mechanism is proposed to mitigate the reliance on pseudo labels and boost the tolerance to noisy labels. 

2. Ensemble labels is adopted as a pseudo label updating strategy to stabilize the training of deep neural networks with noisy labels. 

   [link](https://arxiv.org/pdf/2206.06177.pdf)

## Multimodal Learning with Transformers: A Survey

Check later, [link](https://arxiv.org/pdf/2206.06488.pdf)

## Masked Autoencoders are Robust Data Augmentors

Reconstruction from masked images can be seen as natural data augmentation, as the reconstructed image is distorted. Patches with top-k attention weight in query remain visible.  [link](https://arxiv.org/pdf/2206.04846.pdf)

## Stand-Alone Inter-Frame Attention in Video Models

Video transformers capture motion by spatial-temporal split attention, however, this way may not be optimal because of the movement of the foreground (people) that patches may not belong to the same objects. A better way is to catch local neighboring region inter-frame. [link](https://arxiv.org/pdf/2206.06931.pdf)

## Masked Frequency Modeling for Self-Supervised Visual Pre-Training

(Chen Change Loy, NTU) Masked Frequency Modeling, instead of randomly inserting mask tokens to the input embeddings in the spatial domain, shift the perspective to the frequency domain. Masks out a portion of frequency components of the input image and then predicts the missing frequencies on the frequency spectrum. Due to the spatial redundancy, it may be more ideal to reveal underlying image patterns rather than predicting masked patches in the spatial domain. [link](https://arxiv.org/pdf/2206.07706.pdf)

## Masked Siamese ConvNets

(Yann Lecun, Meta AI) The siamese network encourages embedding to be invariant to distortions, masking is the most general and straightforward method that has the potential to be applied to all kinds of inputs. But masked siamese networks require particular inductive bias and practically only work well with VisionTransformers. Apply a high-pass filter before applying masks, and the model accuracy increases.  [link](https://arxiv.org/pdf/2206.07700.pdf)

## A Simple Data Mixing Prior for Improving Self-Supervised Learning

For mixed images that share the same source, they are semantically related and can be treated as additional positive pairs in self-supervised learning. This method is called "SDMP", which helps to achieve better accuracy and out-of-distribution robustness. [link](https://arxiv.org/pdf/2206.07692.pdf)

## LAVENDER: Unifying Video-Language Understanding as Masked Language Modeling

For each video-language downstream task, LAVENDER unifies them with masked language modeling and uses the same MLM head used in pre-training for all downstream tasks. [link](https://arxiv.org/pdf/2206.07160.pdf)

## It’s Time for Artistic Correspondence in Music and Video

(Carl Vondrick, Columbia University) Given query video or query music, recommending a music or a video. Modeling the long-term temporal context of both video& music signals. (Using InfoNCE, kinda like CLIP pre-training) [link](https://arxiv.org/pdf/2206.07148.pdf)

## MixGen: A New Multi-Modal Data Augmentation

(Mu Li, AWS) Mixup augmentation for image & text pairs. A simple but very strong trick for vision-language pre-training. [link](https://arxiv.org/pdf/2206.08358.pdf)

## Disentangling visual and written concepts in CLIP

(Antonio Torralba, MIT) CLIP has a strong ability to match nonsense words, and the image encoder has an ability to match word images with natural images of scenes described by those words. Author's work is able to cleanly sepatrate spelling capabilities of CLIP from the visual processing of natural images. [link](https://arxiv.org/pdf/2206.07835.pdf)

## OmniMAE: Single Model Masked Pretraining on Images and Videos

(FAIR) Transformer can provide a single unified model for multiple visual modalities. But prior attempts at unified modeling typically use architectures tailored for vision tasks or obtain worse performance compared to single modality models. In this work, authors show that masked autoencoding can be used to train a simple Vision Transformer on images and videos and the single model learns visual representations that are comparable or even better than single-modality representations on both image and video benchmarks. [link](https://arxiv.org/pdf/2206.08356.pdf)

## Beyond Supervised vs. Unsupervised: Representative Benchmarking and Analysis of Image Representation Learning

Different self-supervised learning evaluation benchmarks such as linear probe, NN, K-Means leads to different results. And as single benchmark does not tell the whole story, authors propose two novel metrics: Nearest neighbor graph similarity and linear prediction overlap. [link](https://arxiv.org/pdf/2206.08347.pdf)

## iBoot: Image-bootstrapped Self-Supervised Video Representation Learning

Due to the lack of amount of video datasets, and the prohibitive computation overhead of video datasets. Directly learning self-supervised representations from video data might result in sub-optimal performance. Utilize a image-based model pre-trained with self- or language supervision enables the model to learn strong spatial and temporal information without relying on the video labeled data. 

iBoot architecture: A frozen Image-based target network and a 3D ResNet online network with a linear mapping head. (Quite like BYOL without EMA) [link](https://arxiv.org/pdf/2206.08339.pdf)

## SimA: Simple Softmax-free Attention for Vision Transformers

Transformer is difficult to deploy in many applications, and it's partly because due to the Softmax layer. Authors propose a novel Softmax-free attention block that normalizes query and key matrices with simple l1 norm instead of Softmax layer. Moreover, changing SimA from multi-head to single-head has only a small effect on the accuracy, which may simplify the attention block further. [link](https://arxiv.org/pdf/2206.08898.pdf)

## UNIFIED-IO: A UNIFIED MODEL FOR VISION, LANGUAGE, AND MULTI-MODAL TASKS

Unified model for a bunch of downstream tasks, the overall architechture is Transformer Encoder - Decoder and for different input and tasks using different additional encoder & decoder. Note that Image serialization using a VQ-VAE. [link](https://arxiv.org/pdf/2206.08916.pdf)

## VLMixer: Unpaired Vision-Language Pre-training via Cross-Modal CutMix

Propose a cross-modal cutmix method for implicit cross-modal alignment learning in unpaired VLP. Attaching cross-modal noise on uni-modal data could guide models to learn token-level interactions across modalities for better denoising. Move image tokens to Language tokens.  [link](https://arxiv.org/pdf/2206.08919.pdf)

## Bridge-Tower: Building Bridges Between Encoders in Vision-Language Representation Learning

Existing vision-language models either use lightweight uni-modal encoders and learn to extract, align and fuse both modalities simultaneously in a cross-modal encoder, or feed the last-layer unimodal features directly into the top cross-modal encoder, ignoring the semantic information at the different levels in the deep uni-modal encoders. Authors propose the Bridge-Tower that consists of a visual encoder, a textual encoder, a cross-modal encoder, and multiple lightweight bridge layers. [link](https://arxiv.org/pdf/2206.08657.pdf)

## Crafting Better Contrastive Views for Siamese Representation Learning

Random cropping may not the best choise for contrastive learning:

1. False-positive: object vs. background
2. Trivial pair: too similar for optimization

Propose a novel data augmentation method called contrastive crop: Take semantic information into account and increase variance between positive views. Generate a bounding box of the object from the heatmap. And use the bounding box as a guidance to generate crops. Center-suppressed Sampling: Lower the probability near the center, higher probability at other positions. Larger sampling variance leads to smaller overlap and less similarity. [link](https://openaccess.thecvf.com/content/CVPR2022/papers/Peng_Crafting_Better_Contrastive_Views_for_Siamese_Representation_Learning_CVPR_2022_paper.pdf)

## Vicinity Vision Transformer

For each image patch, the attention weight is adjust based on its 2D Manhattan distance measured by its neighboring patches. In this case, the neighbouring patches will receive stronger attention than far away patches. [link](https://arxiv.org/pdf/2206.10552.pdf)

## Few-Max: Few-Shot Domain Adaptation for Unsupervised Contrastive Representation Learning

The performance of contrastive learning may degrade due to the lack of target datasets, especially for few-shot learning cases. Uses ideas from knowledge distillation and sim2real generation methods to learn a novel representation using a pretrained network as a regularizer.  [link](https://arxiv.org/pdf/2206.10137.pdf)

## CLiMB: A Continual Learning Benchmark for Vision-and-Language Tasks

Existing CL benchmarks have facilitated research on task adaptation and mitigating catastrophic forgetting, but are limited to vision-only and language-only tasks. Authors propose CLiMB to study the challenge of learning multimodal tasks in a CL setting, and to systematically evaluate how upstream continual learning can rapidly generalize to new multimodal tasks in a CL setting. 

CliMB evaluates candidate CL models and learning algorithms in two phases, for phase 1, upstream continual learning: a pre-trained multimodal model is trained on a sequence of vision and language tasks, and evaluated after each task on its degree of Forgetting of past tasks and knowledge transfer to the next task. For phase 2, after each multimodal task the model is evaluated for its downstream Low-Shot transfer capability on both multimodal and unimodal tasks. [link](https://arxiv.org/pdf/2206.09059.pdf)

## Bi-Calibration Networks for Weakly-Supervised Video Representation Learning

Authors propose a new design of mutual calibration between query and text to boost weakly-supervised video representation learning. Couples two calibrations to learn the amendment from text to query and vice versa. Executes clustering on all the titles of the videos searched by an identical query and takes the centroid of each cluster as a text prototype. [link](https://arxiv.org/pdf/2206.10491.pdf)

## Automatic Concept Extraction for Concept Bottleneck-based Video Classification

To decouple the visual representation via concept guides. For complex tasks, the labels and the relationship between visual elements span many frames. So automatically discovering concepts and extracting them is very important. CoDEx identifies a rich set of complex concept abstractions from NL explanations of videos and obviates the need to predefine the amorphous set of concepts. [link](https://arxiv.org/pdf/2206.10129.pdf)

## TiCo: Transformation Invariance and Covariance Contrast for Self-Supervised Visual Representation Learning

(Yann LeCun, MAIR) Based on maximizing the agreement among embeddings of different distorted versions of the same image, which pushes the encoder to produce transformation invariant representations. To avoid trivial solutions, regularize the covariance matrix of the embeddings from different images by penalizing low-rank solutions. Can be seen as a variant of MoCo with an implicit memory bank (regularizing the covariance matrix) without additional memory cost. (Barlow-twins) [link](https://arxiv.org/pdf/2206.10698.pdf)

## Siamese Contrastive Embedding Network for Compositional Zero-Shot Learning

Compositional zero-shot learning: Recognize unseen compositions formed from seen states and objects during training. Existing methods recognize state and object with two separate classifiers, ignoring the impact of the interaction between seen and unseen composition sets. Other methods try to learn the joint representation of the state-object compositions, but the domain gap is still a severe problem. 

Authors embed the visual feature into a siamese contrastive space to capture **prototypes** of them separately to alleviate the interaction between state and object. A novel module called state transition module is proposed to increase the diversity of training compositions. Firstly, the visual features are projected into state/object-based contrastive spaces to gain the prototypes of state and object. Then, to excavate the discriminative prototypes by contrastive constraints, set up specific databases as positive samples. And a shared irrelevant database is built up as a negative sample set.  [link](https://arxiv.org/pdf/2206.14475.pdf)

## Parameter-Efficient Image-to-Video Transfer Learning

(Hongsheng Li, CUHK) Existing image-to-video transfer learning strategies are typically under a full fine-tuning setting, which is parameter-inefficient since a specific instance of such a large model is resulted for each downstream task. Authors propose to only train a lightweight Spatio-temporal adapter with much fewer parameters for each individual downstream task at a significantly smaller computational cost. 

The module contains only a pair of up & sub sampling and a depthwise-3D convolution layer. (I guess there's a typo in Fig.1.) [link](https://arxiv.org/pdf/2206.13559.pdf)

## SLIC: Self-Supervised Learning with Iterative Clustering for Human Action Videos

Clustering-based video self-supervised learning. 83.2 top1 accuracy on UCF101. The current video self-supervised learning method is done conservatively to avoid false positives.  A typical assumption is that similar clips only occur temporally close within a single video, leading to insufficient examples of motion similarity. Authors propose an iterative clustering method to group similar video instances. This enables to leverage pseudo-labels from the cluster assignments to sample harder positives and negatives. 

Using a triplet margin loss and acquiring pseudo-labels from the cluster assignments to sample triplets. The total loss consists of a temporal discrimination loss and an instance-based triplet loss. 

1. Iterative Clustering

   Adopt the FINCH algorithm to obtain pseudo-labels from clustering the video embeddings, which could discover groupings in the data by linking the first neighbor relations of each sample, and hence does not require any prior knowledge of the data distribution. Concretely, FINCH computes the first neighbor for each video instance in the feature space using the cosine distance metric. Then FINCH will generate an adjacency link matrix that links each video instance to its first neighbors.

2. Instance-based Triplet Loss

   To sample harder positives and negatives during training. Hard positives are defined as different videos from the same semantic class, while hard negatives are defined as samples that are closer to the anchor than the positive. Also, multi-view positive is adopted as optical flow is used as the second view in addition to RGB. The RGB clip will be replaced with optical flow with a probability.

3. Temporal Discrimination Loss

   Given an anchor clip x, the positive clip is designated as the spatial augmentation of the anchor clip. The negative is any temporally non-overlapping clip from the same instance or from a different instance in the same cluster. 

Limitation: it's sensitive to the false positive and false negative rate.

## On-Device Training Under 256KB Memory

(Ji Lin, Song Han, MIT) Very cool stuff. Including quantization-aware scaling, sparse update, tiny training engine that enables lifelong learning on an STM32 chip. [link](https://arxiv.org/pdf/2206.15472.pdf)

## Exploring Temporally Dynamic Data Augmentation for Video Recognition

Few existing augmentation recipes for video recognition naively extend the image augmentation methods by applying the same operations to the whole video frames. Authors propose DynaAugment. The magnitude of augmentation operations on each frame is changed by an effective mechanism. 

Based on Fourier analysis, an arbitrary signal can be decomposed into multiple basis functions. All of the temporal variations can be represented by the random weighted sum of diverse-frequency sinusoidal basis functions. Temporal variations are defined as geometrically and photometrically. In a word, the magnitude of the augmentation is ever-changing and follows a Fourier way. Boosts the performance about 1.7% for SlowFast. 

Limitation: Lack of finding these recipes combined with other augmentation and regularization methods. [link](https://arxiv.org/pdf/2206.15015.pdf)

## ReCo: Retrieve and Co-segment for Zero-shot Transfer

(Weidi, SJTU) Curate training sets from unlabelled images by leveraging the retrieval abilities of CLIP. 

The overall workflow is:

1. Retrieval 

   Given a large-scale unlabelled dataset and a category to segment, curate an archive of k images from the unlabelled dataset using CLIP.

2. Co-segmentation

   Using a pre-trained visual encoder to extract dense features from the archive images, which are used to generate a reference image embedding for the given category via a co-segmentation process.

3. Inference

   During inference, the reference image is employed to produce an initial segmentation of the target concept which is refined with DenseCLIP.

[link](https://arxiv.org/pdf/2206.07045.pdf)

## DenseCLIP: Language-Guided Dense Prediction with Context-Aware Prompting

(Yongming Rao, THU) To better leverage the pre-trained knowledge from CLIP, convert the original image-text matching problem in CLIP to a pixel-text matching problem and use the pixel-text score maps to guide the learning of dense prediction models. (Detection, segmentation, etc) [link](https://openaccess.thecvf.com/content/CVPR2022/papers/Rao_DenseCLIP_Language-Guided_Dense_Prediction_With_Context-Aware_Prompting_CVPR_2022_paper.pdf)

## Exploiting Transformation Invariance and Equivariance for Self-supervised Sound Localisation

(Weidi) The composition of data augmentations plays a critical role in sound localizing tasks. Also enforcing geometric consistency substantially improves the quality of learned representations (the detected source should follow the same transformation applied on input video frames i.e. transformation equivariance). 

The main difference between this work and LVS [Honglie Chen et al] is to explore the impact of image data augmentation on audio localisation tasks. The overall framework is a siamese network with two identical branches. As for audio augmentations, randomly masking a period of the audio spectrogram. And for visual frames, the transformations were split into two groups: appearance transformations (gaussian, jittering, grayscale) and geometrical transformations(geometrical shapes and locations). [link](https://arxiv.org/pdf/2206.12772.pdf)

## RegMixup: Mixup as a Regularizer Can Surprisingly Improve Accuracy and Out Distribution Robustness

Traditional mixup may have several problems: 1. Entropy is higher (due to the smoothed label) 2. The model has only seen mixed labels. 3. The augmentation is not strong enough

Authors propose RegMixup, an explicit assembly of ERM and VRM-based approximations to data distribution. Actually, this algorithm just applies mixup as a regularization term to fix the overall Cross-Entropy loss. Besides, the basic mixup applies Beta distribution with parameter α=0.1, and RegMixup applies α=10. Generates more interpolated data. [link](https://arxiv.org/abs/2206.14502)

## Mixed Sample Data Augmentation

1. **Gridmask data augmentation**

   Grid-wise data cutout.

2. **SmoothMix: a Simple Yet Effective Data Augmentation to Train Robust Classifiers**

   Instead of straight forward cutmix, apply a gaussian kernel to smooth the mix boundary.

3. **FMix: Enhancing Mixed Sample Data Augmentation**

   The augmentation region is not square. Apply more generated mixup square. (Saliency map way & tradition way)

   Apply a Fourier transformation and a low-pass filter to generate the mask. (Low-frequency regions could infers those consistency region)

4. **Manifold Mixup: Better Representations by Interpolating Hidden States**

   Apply mixup at latent space. 

5. **AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty**

   Prior data augmentation is in a chain. (Polarize -> rotation -> jittering ->....) A better way is mixing up shorter chains. Jensen-Shannon Divergence Consistency Loss is applied to guarantee consistency in the latent space for different augmented views. 

6. **Un-Mix: Rethinking Image Mixtures for Unsupervised Visual Representation Learning**

   In the case of contrastive learning, the training signal is very strong to divide the positive and negetive samples to the contrastive decision boundary (the margin is huge), which is 'over-confidence' and may lead to performance degradation. 

   Without a memory bank: I1 & I2 is defined as a different image for mixing. Inverse the I2 list to mix with I1. Then the distance between I1 & I2 is defined as mixup rate lambda. 

   With a memory bank: Negative pairs can be {original, original}, {original, mixed}, {mixed, mixed}. One memory bank with the representations from original/unmixed images is enough to obtain good performance. 

7. **CropMix: Sampling a Rich Input Distribution via Multi-Scale Cropping**

   For an original image, utilizing multiple cropping operations with distinct crop scales to obtain multiple cropped views. 

8. **KeepAugment: A Simple Information-Preserving Data Augmentation Approach**

   Apply a saliency map measuring method to always keep the important regions. 

   To speed up saliency, there are basically 2 ways. First is to using a relatively low resolution to boost the speed. Second is to use a cheap network to generate saliency map.

9. **Attentive CutMix: An Enhanced Data Augmentation Approach for Deep Learning Based Image Classification**

    Utilizing the top 6 attentive patches from a 7*7 grid to generate saliency patches. 

10. **SaliencyMix: A Saliency Guided Data Augmentation Strategy for Better Regularization**

    Select the most salient part of the image to apply the Cutmix method.

11. **Puzzle Mix: Exploiting Saliency and Local Statistics for Optimal Mixup**

    To keep both of the most salient parts in the Cutmix image pairs, transport the axis of the image patch.

12. **TransMix: Attend to Mix for Vision Transformers**

    The original definition of labels in Cutmix data augmentation is using the area ratio. TransMix utilizes the saliency to define labels. 

## SeCo: Exploring Sequence Supervision for Unsupervised Representation Learning

Materialize the supervisory signals through determining whether a pair of samples is from one frame or from one video and whether a triplet of samples is in the correct temporal order. Given an unlabeled video sequence v, firstly sample three frames randomly and take the first frame s in time order as the anchor frame. 

There are 3 pretext tasks in SeCo. 

1. Inter-frame instance discrimination

   Determine whether two frame patches are from the same video. Define all the keys within the same video as positive ones, and the frame patches sampled from other videos in neighboring batches are taken as the negative keys. The objective function in this task is defined as the averaged sum of all the contrastive losses with regard to each positive q-k pair. 

2. Intra-frame instance discrimination

   Distinguish the frame patches of the same frame from the ones of the other frames in a video. Directly boost the spatial perspective. 

3. Temporal order validation

   Whether a series of frame patches are in the correct temporal order. 

## SELF-SUPERVISED LEARNING FOR VIDEOS: A SURVEY

(Mubarak Shah, UCF) The self-supervised learning for videos can be summarized into three different categories based on their learning objectives: 1) pre-text tasks, 2) generative modeling, 3) contrastive learning.

Video has the property that it always comes with audio, and text (speak, captions). The motion could also be seen as a modality. 

Downstream tasks: 

1. action recognition (action classification). Metrics: acc, prec, recall.
2. action segmentation: temporal localization, find the start and end time of an action and also the type of action in a video clip. Metrics: mAP, frame accuracy, recall. Recall focuses on the proposal of an action during a time segment and how accurate it is compared to the ground truth and is used for temporal step localization. UIsing mAP utilizes an intersection-over-union for the time dimension (t-IOU) and measures the mean of the average accuracy of all the predicted proposals for all classes. FA measures the accuracy of the model when predicting the action frame-by-frame. 
3. video retrieval: find similar videos given a query video. Metric: standard retrieval metrics (retrieval, recall, similarity measures). Recall is the fraction of the videos returned that are successfully retrieved that are relevant to the query video (top k). 
4. text-to-video retrieval: given a text query as input, return corresponding matching videos. 
5. video captioning: generative task, given an input video clip, return the caption of the clip. 

Pretext learning:

1. Appearance statistics prediction

   Predict or classify an appearance modifying augmentation applied to a clip (Color, rotation, noise). 

2. Playback speed - perform the best

   Modifying the frame selection in a way that the playback speed is altered. Collecting every p frames where p is the playback rate, either speeding up the video or slow it down. Type & speed. 

3. Temporal order classification

   Each video V is split into clips of t frames from the total T frames. Correct order or shuffled order. Clip based or frame based. 

4. Video Jigsaw

   Number of patches that are permuted, leading to too many permutations to select from.

5. Masked Modeling

   Both vidual and text signals. Synchronized from the same video or shifted several seconds. 

Masked modelling should be further investigated in future work.

Generative Approaches:

1. Adversarial and reconstruction

