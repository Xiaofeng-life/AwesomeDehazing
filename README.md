# A Collection of DL-based Dehazing Methods

This repository provides a summary of deep learning based dehazing algorithms. 
Classification is based on our review paper, "A Comprehensive Survey and Taxonomy on Single Image Dehazing Based on Deep Learning", whic has been accepted by ACM Computing Surveys. Since this repository involves a lot of professional vocabulary, it is recommended to read our review paper before using this repository.
If you find these codes useful, we appreciate it very much if you can cite our paper: https://dl.acm.org/doi/10.1145/3576918

@article{gui2023comprehensive,

&emsp  title={A comprehensive survey and taxonomy on single image dehazing based on deep learning},
  
&emsp  author={Gui, Jie and Cong, Xiaofeng and Cao, Yuan and Ren, Wenqi and Zhang, Jun and Zhang, Jing and Cao, Jiuxin and Tao, Dacheng},
  
&emsp  journal={ACM Computing Surveys},
  
  volume={55},
  
  number={13s},
  
  pages={1--37},
  
  year={2023},
  
  publisher={ACM New York, NY}
  
}

We classify dehazing algorithms into supervised, semi-supervised and unsupervised. 
See our paper for more details.
If you have any suggestions, feel free to contact me (Email: cxf_svip@163.com). Thanks.

[Supervised Dehazing Methods](#supervised)

 * [1. Learning of t(x)](#s-1)

 * [2. Joint Learning of t(x) and A](#s-2)

 * [3. Non-explicitly embedded ASM](#s-3)

 * [4. Generative adversarial network](#s-4)

 * [5. Level-aware](#s-5)

 * [6. Multi-function fusion](#s-6)

 * [7. Transformation and decomposition of input](#s-7)

* [8. Knowledge distillation](#s-8)

* [9. Transformation of colorspace](#s-9)

* [10. Contrastive learning](#s-10)

* [11. Non-deterministic output](#s-11)

* [12. Retinex model](#s-12)

* [13. Residual learning](#s-13)

* [14. Frequency domain](#s-14)

* [15. Joint dehazing and depth estimation](#s-15)

* [16. Detection and segmentation with dehazing](#s-16)

* [17. End-to-end CNN](#s-17)

[Semi-supervised Dehazing Methods](#semi-supervised)

* [1. Pretrain backbone and fine-tune](#ss-1)

* [2. Disentangled and reconstruction](#ss-2)

* [3. Two-branches training](#ss-3)

<p id="supervised"></p>

[Unsupervised Dehazing Methods](#unsupervised)

* [1. Unsupervised domain translation](#us-1)

* [2. Learning without haze-free images](#us-2)

* [3. Unsupervised image decomposition](#us-3)

* [4. Zero-Shot Image Dehazing](#us-4)


[Hazy Dataset](#dataset)

## Supervised Dehazing Methods

<p id="s-1"></p>

### 1. Learning of t(x)
* Dehazenet: An end-to-end system for single image haze removal.
\[[paper](https://ieeexplore.ieee.org/abstract/document/7539399)\]
\[[code](https://github.com/caibolun/DehazeNet)\]

* ABC-NET: Avoiding Blocking Effect & Color Shift Network for Single Image Dehazing Via Restraining Transmission Bias.
\[[paper](https://ieeexplore.ieee.org/abstract/document/9190777)\]
\[code\]

* MSCNN: Single Image Dehazing via Multi-scale Convolutional Neural Networks.
\[[paper](https://linkspringer.53yu.com/chapter/10.1007/978-3-319-46475-6_10)\]
\[[code](https://github.com/rwenqi/Multi-scale-CNN-Dehazing)\]

* MSCNN-HE: Single Image Dehazing via Multi-scale Convolutional Neural Networks with Holistic Edges.
\[[paper](https://linkspringer.53yu.com/article/10.1007/s11263-019-01235-8)\]
\[code\]

* SID-JDM: SINGLE IMAGE DEHAZING VIA A JOINT DEEP MODELING.
\[[paper](https://ieeexplore.ieee.org/abstract/document/8451663)\]
\[code\]

* LATPN: Learning Aggregated Transmission Propagation Networks for Haze Removal and Beyond.
\[[paper](https://ieeexplore.ieee.org/abstract/document/8450630)\]
\[code\]

<p id="s-2"></p>

### 2. Joint Learning of t(x) and A

* DCPDN: Densely Connected Pyramid Dehazing Network.
\[[paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_Densely_Connected_Pyramid_CVPR_2018_paper.html)\]
\[[code](https://github.com/hezhangsprinter/DCPDN)\]

* DSIEN: Dense Scene Information Estimation Network for Dehazing.
\[[paper](https://openaccess.thecvf.com/content_CVPRW_2019/html/NTIRE/Guo_Dense_Scene_Information_Estimation_Network_for_Dehazing_CVPRW_2019_paper.html)\]
\[[code](https://github.com/tT0NG/AtJ-DH)\]

* LDPID: Learning Deep Priors for Image Dehazing.
\[[paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Liu_Learning_Deep_Priors_for_Image_Dehazing_ICCV_2019_paper.html)\]
\[[code](https://github.com/lewisyangliu/LDP)\]

* PMHLD: Patch Map-Based Hybrid Learning DehazeNet for Single Image Haze Removal.
\[[paper](https://ieeexplore.ieee.org/abstract/document/9094006)\]
\[[code](https://github.com/weitingchen83/Dehazing-PMHLD-Patch-Map-Based-Hybrid-Learning-DehazeNet-for-Single-Image-Haze-Removal-TIP-2020)\]

* HRGAN: Visual Haze Removal by a Unified Generative Adversarial Network.
\[[paper](https://ieeexplore.ieee.org/abstract/document/8529212)\]
\[code\]

<p id="s-3"></p>

### 3. Non-explicitly embedded ASM

* AOD-Net: All-In-One Dehazing Network.
\[[paper](https://openaccess.thecvf.com/content_iccv_2017/html/Li_AOD-Net_All-In-One_Dehazing_ICCV_2017_paper.html)\]
\[[code](https://github.com/weber0522bb/AODnet-by-pytorch)\]

* DehazeGAN: When Image Dehazing Meets Differential Programming.
\[[paper](http://www.pengxi.me/wp-content/uploads/Papers/2018-IJCAI-DehazeGAN.pdf)\]
\[code\]

* PFDN: Physics-Based Feature Dehazing Networks.
\[[paper](https://linkspringer.53yu.com/chapter/10.1007/978-3-030-58577-8_12)\]
\[code\]

* SI-DehazeGAN: Single-Image Dehazing via Compositional Adversarial Network.
\[[paper](https://ieeexplore.ieee.org/abstract/document/8946591)\]
\[code\]

<p id="s-4"></p>

### 4. Generative adversarial network

* EPDN: Enhanced Pix2pix Dehazing Network.
\[[paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Qu_Enhanced_Pix2pix_Dehazing_Network_CVPR_2019_paper.html)\]
\[[code](https://github.com/ErinChen1/EPDN)\]

* PGC-UNet: Pyramid Global Context Network for Image Dehazing.
\[[paper](https://ieeexplore.ieee.org/abstract/document/9252912)\]
\[[code](https://github.com/phoenixtreesky7/PGC-DN)\]

* RI-GAN: An End-To-End Network for Single Image Haze Removal
\[[paper](https://openaccess.thecvf.com/content_CVPRW_2019/html/NTIRE/Dudhane_RI-GAN_An_End-To-End_Network_for_Single_Image_Haze_Removal_CVPRW_2019_paper.html)\]
\[code\]

* DHGAN: High-Resolution Image Dehazing With Respect to Training Losses and Receptive Field Sizes.
\[[paper](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w13/html/Sim_High-Resolution_Image_Dehazing_CVPR_2018_paper.html)\]
\[code\]

* SA-CGAN: Scale-aware Conditional Generative Adversarial Network for Image Dehazing.
\[[paper](https://openaccess.thecvf.com/content_WACV_2020/html/Sharma_Scale-aware_Conditional_Generative_Adversarial_Network_for_Image_Dehazing_WACV_2020_paper.html)\]
\[code\]

<p id="s-5"></p>

### 5. Level-aware

* LAP-Net: Level-Aware Progressive Network for Image Dehazing
\[[paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Li_LAP-Net_Level-Aware_Progressive_Network_for_Image_Dehazing_ICCV_2019_paper.html)\]
\[code\]

<p id="s-6"></p>

### 6. Multi-function fusion

* DMMFD: Deep Multi-Model Fusion for Single-Image Dehazing.
\[[paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Deng_Deep_Multi-Model_Fusion_for_Single-Image_Dehazing_ICCV_2019_paper.html)\]
\[[code](https://github.com/zijundeng/DM2F-Net)\]

<p id="s-7"></p>

### 7. Transformation and decomposition of input

* GFN: Gated Fusion Network for Single Image Dehazing.
\[[paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Ren_Gated_Fusion_Network_CVPR_2018_paper.html)\]
\[[code](https://github.com/rwenqi/GFN-dehazing)\]

* MSRL-DehazeNet: Multi-Scale Deep Residual Learning-Based Single Image Haze Removal via Image Decomposition.
\[[paper](https://ieeexplore.ieee.org/abstract/document/8931240)\]
\[code\]

* DPDP-Net: Dual-Path in Dual-Path Network for Single Image Dehazing.
\[[paper](https://www.ijcai.org/Proceedings/2019/0643.pdf)\]
\[code\]

* DIDH: Towards domain invariant single image dehazing.
\[[paper](https://www.aaai.org/AAAI21Papers/AAAI-1706.ShyamP.pdf)\]
\[[code](https://github.com/PS06/DIDH)\]

<p id="s-8"></p>

### 8. Knowledge distillation

* KDDN: Distilling Image Dehazing With Heterogeneous Task Imitation.
\[[paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Hong_Distilling_Image_Dehazing_With_Heterogeneous_Task_Imitation_CVPR_2020_paper.html)\]
\[[code](https://github.com/FadeoN/Distilling-Image-Dehazing-With-Heterogeneous-Task-Imitation)\]

* KTDN: Knowledge Transfer Dehazing Network for NonHomogeneous Dehazing.
\[[paper](https://openaccess.thecvf.com/content_CVPRW_2020/html/w31/Wu_Knowledge_Transfer_Dehazing_Network_for_NonHomogeneous_Dehazing_CVPRW_2020_paper.html)\]
\[[code](https://github.com/GlassyWu/KTDN)\]

* SRKTDN: Applying Super Resolution Method to Dehazing Task.
\[[paper](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/html/Chen_SRKTDN_Applying_Super_Resolution_Method_to_Dehazing_Task_CVPRW_2021_paper.html)\]
\[code\]

* DALF: A guiding teaching and dual adversarial learning framework for a single image dehazing.
\[[paper](https://linkspringer.53yu.com/article/10.1007/s00371-021-02184-5)\]
\[[code](None)\]

<p id="s-9"></p>

### 9. Transformation of colorspace

* AIPNet: Image-to-Image Single Image Dehazing With Atmospheric Illumination Prior.
\[[paper](https://ieeexplore.ieee.org/abstract/document/8454467)\]
\[code\]

* MSRA-Net: Multi-scale residual attention network for single image dehazing.
\[[paper](https://www.sciencedirect.com/science/article/pii/S1051200421003663)\]
\[code\]

* TheiaNet: Towards fast and inexpensive CNN design choices for image dehazing.
\[[paper](https://www.sciencedirect.com/science/article/pii/S1047320321000791)\]
\[code\]

* RYF-Net: Deep Fusion Network for Single Image Haze Removal.
\[[paper](https://ieeexplore.ieee.org/abstract/document/8802288)\]
\[code\]

<p id="s-10"></p>

### 10. Contrastive learning

* AECR-Net:Contrastive Learning for Compact Single Image Dehazing.
\[[paper](https://openaccess.thecvf.com/content/CVPR2021/html/Wu_Contrastive_Learning_for_Compact_Single_Image_Dehazing_CVPR_2021_paper.html)\]
\[[code](https://github.com/GlassyWu/AECR-Net)\]

<p id="s-11"></p>

### 11. Non-deterministic output

* pWAE: Pixel-Wise Wasserstein Autoencoder for Highly Generative Dehazing.
\[[paper](https://ieeexplore.ieee.org/abstract/document/9447190)\]
\[code\]

* DehazeFlow: Multi-scale Conditional Flow Network for Single Image Dehazing
\[[paper](https://dl.acm.org/doi/abs/10.1145/3474085.3475432)\]
\[[code](https://github.com/iCVTEAM/DehazeFlow)\]

<p id="s-12"></p>

### 12. Retinex model

* RDN: Deep Retinex Network for Single Image Dehazing.
\[[paper](https://ieeexplore.ieee.org/abstract/document/9274531)\]
\[code\]

<p id="s-13"></p>

### 13. Residual learning

* GCA-Net: Gated Context Aggregation Network for Image Dehazing and Deraining.
\[[paper](https://ieeexplore.ieee.org/abstract/document/8658661)\]
\[[code](https://github.com/cddlyf/GCANet)\]

* DRL: Recursive Deep Residual Learning for Single Image Dehazing.
\[[paper](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w13/html/Du_Recursive_Deep_Residual_CVPR_2018_paper.html)\]
\[[code](https://github.com/yixindu1573/Recursive-Deep-Residual-Learning-for-Single-Image-Dehazing-DRL)\]

* SID-HL: Single image dehazing based on learning of haze layers.
\[[paper](https://www.sciencedirect.com/science/article/pii/S092523122030028X)\]
\[code\]

* POGAN: Recursive Image Dehazing via Perceptually Optimized Generative Adversarial Network.
\[[paper](https://ieeexplore.ieee.org/abstract/document/9025378)\]
\[[code](https://github.com/yixindu1573/Recursive-Image-Dehazing-via-Perceptually-Optimized-Generative-Adversarial-Network-POGAN)\]

<p id="s-14"></p>

### 14. Frequency domain

* Wavelet U-Net: Wavelet U-Net and the Chromatic Adaptation Transform for Single Image Dehazing.
\[[paper](https://ieeexplore.ieee.org/abstract/document/8803391)\]
\[[code](https://github.com/dectrfov/Wavelet-U-net-Dehazing)\]

* MsGWN: Deep multi-scale gabor wavelet network for image restoration.
\[[paper](https://ieeexplore.ieee.org/abstract/document/9053804)\]
\[code\]

* EMRA-Net: An ensemble multi-scale residual attention network (EMRA-net) for image Dehazing.
\[[paper](https://link.springer.com/article/10.1007/s11042-021-11081-x)\]
\[[code](https://github.com/Maverick-3/EMRA-Net)\]

* TDN: Trident dehazing network.
\[[paper](https://openaccess.thecvf.com/content_CVPRW_2020/html/w31/Liu_Trident_Dehazing_Network_CVPRW_2020_paper.html)\]
\[[code](https://github.com/lj1995-computer-vision/Trident-Dehazing-Network)\]

* DW-GAN: A Discrete Wavelet Transform GAN for NonHomogeneous Dehazing.
\[[paper](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/html/Fu_DW-GAN_A_Discrete_Wavelet_Transform_GAN_for_NonHomogeneous_Dehazing_CVPRW_2021_paper.html)\]
\[[code](https://github.com/houjie8888/dehaze/tree/d06cb24e9baf35ff8bde630f7f5080d27df9f7df/DW-GAN-Dehazing-main/DW-GAN-Dehazing-main)\]

<p id="s-15"></p>

### 15. Joint dehazing and depth estimation

* SDDE: CNN-Based Simultaneous Dehazing and Depth Estimation
\[[paper](https://ieeexplore.ieee.org/abstract/document/9197358)\]
\[code\]

* S2DNet: Depth Estimation From Single Image and Sparse Samples.
\[[paper](https://ieeexplore.ieee.org/abstract/document/9040600)\]
\[code\]

* DDRL: Reinforced Depth-Aware Deep Learning for Single Image Dehazing.
\[[paper](https://ieeexplore.ieee.org/abstract/document/9054504)\]
\[[code](http://signal.ee.psu.edu/research/DDRL.html)\]

* DeAID: Depth aware image dehazing.
\[[paper](https://linkspringer.53yu.com/article/10.1007/s00371-021-02089-3)\]
\[code\]

* TSDCN-Net: Two-Stage Image Dehazing with Depth Information and Cross-Scale Non-Local Attention.
\[[paper](https://ieeexplore.ieee.org/abstract/document/9672042)\]
\[code\]

<p id="s-16"></p>

### 16. Detection and segmentation with dehazing

* LEAAL: Deep Dehazing Network With Latent Ensembling Architecture and Adversarial Learning.
\[[paper](https://ieeexplore.ieee.org/abstract/document/9298479)\]
\[code\]

* SDNet: Semantic-Aware Dehazing Network With Adaptive Feature Fusion.
\[[paper](https://ieeexplore.ieee.org/abstract/document/9622120)\]
\[code\]

* UDnD: Unified Density-Aware Image Dehazing and Object Detection in Real-World Hazy Scenes.
\[[paper](https://openaccess.thecvf.com/content/ACCV2020/html/Zhang_Unified_Density-Aware_Image_Dehazing_and_Object_Detection_in_Real-World_Hazy_ACCV_2020_paper.html)\]
\[[code](https://github.com/xiqi98/UDnD)\]

<p id="s-17"></p>

### 17. End-to-end CNN

* FFA-Net: Feature Fusion Attention Network for Single Image Dehazing.
\[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/6865)\]
\[[code](https://github.com/zhilin007/FFA-Net)\]

* GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing.
\[[paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Liu_GridDehazeNet_Attention-Based_Multi-Scale_Network_for_Image_Dehazing_ICCV_2019_paper.html)\]
\[[code](https://github.com/proteus1991/GridDehazeNet)\]

* SAN: Selective Attention Network for Image Dehazing and Deraining.
\[[paper](https://dl.acm.org/doi/abs/10.1145/3338533.3366688)\]
\[[code](https://github.com/liang233/Selective-Attention-Network-for-Image-Dehazing-and-Deraining)\]

* HFF: Hierarchical Feature Fusion With Mixed Convolution Attention for Single Image Dehazing.
\[[paper](https://ieeexplore.ieee.org/abstract/document/9381290)\]
\[code\]

* 4kDehazing: Ultra-high-definition image dehazing via multi-guided bilateral learning.
\[[paper](https://ieeexplore.ieee.org/abstract/document/9578433)\]
\[[code](https://github.com/zzr-idam/4KDehazing)\]

* CAE: Convolutional Autoencoder For Single Image Dehazing.
\[[paper](http://www.lai-online.net/edmund/Publications/c/ICIP2019_Chen_Lai.pdf)\]
\[code\]

* DESU: Image Dehazing With Contextualized Attentive U-NET.
\[[paper](https://ieeexplore.ieee.org/abstract/document/9190725)\]
\[[code](https://github.com/yeanwei97/DSEU)\]

* 123-CEDH: Dense `123' Color Enhancement Dehazing Network.
\[[paper](https://openaccess.thecvf.com/content_CVPRW_2019/html/NTIRE/Guo_Dense_123_Color_Enhancement_Dehazing_Network_CVPRW_2019_paper.html)\]
\[[code](https://github.com/tT0NG/123-CEDH)\]

* MSBDN: Multi-Scale Boosted Dehazing Network With Dense Feature Fusion.
\[[paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Dong_Multi-Scale_Boosted_Dehazing_Network_With_Dense_Feature_Fusion_CVPR_2020_paper.html)\]
\[[code](https://github.com/BookerDeWitt/MSBDN-DFF)\]

* DMHN: Fast Deep Multi-Patch Hierarchical Network for Nonhomogeneous Image Dehazing.
\[[paper](https://openaccess.thecvf.com/content_CVPRW_2020/html/w31/Das_Fast_Deep_Multi-Patch_Hierarchical_Network_for_Nonhomogeneous_Image_Dehazing_CVPRW_2020_paper.html)\]
\[[code](https://github.com/diptamath/Nonhomogeneous_Image_Dehazing)\]




<!--################################################################-->

<p id="semi-supervised"></p>

## Semi-supervised Dehazing Methods

<p id="ss-1"></p>

### 1. Pretrain backbone and fine-tune

* PSD: Principled synthetic-to-real dehazing guided by physical priors.
\[[paper](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_PSD_Principled_Synthetic-to-Real_Dehazing_Guided_by_Physical_Priors_CVPR_2021_paper.html)\]
\[[code](https://github.com/zychen-ustc/PSD-Principled-Synthetic-to-Real-Dehazing-Guided-by-Physical-Priors)\]

* SSDT: Single Image Dehazing via Semi-Supervised Domain Translation and Architecture Search.
\[[paper](https://ieeexplore.ieee.org/abstract/document/9573285)\]
\[[code](https://github.com/jklp2/SIDSDT)\]

<p id="ss-2"></p>

### 2. Disentangled and reconstruction

* DCNet: Dual-Task Cycle Network for End-to-End Image Dehazing.
\[[paper](https://ieeexplore.ieee.org/abstract/document/9428282)\]
\[code\]

* FSR: From Synthetic to Real: Image Dehazing Collaborating with Unlabeled Real Data.
\[[paper](https://dl.acm.org/doi/abs/10.1145/3474085.3475331)\]
\[[code](https://github.com/liuye123321/DMT-Net)\]

* CCDM: Color-Constrained Dehazing Model
\[[paper](https://openaccess.thecvf.com/content_CVPRW_2020/html/w51/Zhang_Color-Constrained_Dehazing_Model_CVPRW_2020_paper.html)\]
\[code\]

<p id="ss-3"></p>

### 3. Two-branches training

* DAID: Domain Adaptation for Image Dehazing.
\[[paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Shao_Domain_Adaptation_for_Image_Dehazing_CVPR_2020_paper.html)\]
\[[code](https://github.com/HUSTSYJ/DA_dahazing)\]

* SSID: Semi-Supervised Image Dehazing.
\[[paper](https://ieeexplore.ieee.org/abstract/document/8902220)\]
\[[code](https://github.com/Prevalenter/semi-dehazing)\]

* SSIDN: Semi-Supervised image dehazing network.
\[[paper](https://linkspringer.53yu.com/article/10.1007/s00371-021-02265-5)\]
\[code\]




<!--################################################################-->

<p id="unsupervised"></p>

## Unsupervised Dehazing Methods

<p id="us-1"></p>

### 1. Unsupervised domain translation

* Cycle-Dehaze: Enhanced CycleGAN for Single Image Dehazing.
\[[paper](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w13/html/Engin_Cycle-Dehaze_Enhanced_CycleGAN_CVPR_2018_paper.html)\]
\[[code](https://github.com/engindeniz/Cycle-Dehaze)\]

* CDNet: Single Image De-Hazing Using Unpaired Adversarial Training.
\[[paper](https://ieeexplore.ieee.org/abstract/document/8658408)\]
\[code\]

* E-CycleGAN: End-to-End Single Image Fog Removal Using Enhanced Cycle Consistent Adversarial Networks.
\[[paper](https://ieeexplore.ieee.org/abstract/document/9139368)\]
\[code\]

* USID: Towards Unsupervised Single Image Dehazing With Deep Learning.
\[[paper](https://ieeexplore.ieee.org/abstract/document/8803316)\]
\[code\]

* DCA-CycleGAN: Unsupervised single image dehazing using Dark Channel Attention optimized CycleGAN.
\[[paper](https://www.sciencedirect.com/science/article/pii/S1047320321002923)\]
\[code\]

* DHL-Dehaze: Discrete Haze Level Dehazing Network.
\[[paper](https://dl.acm.org/doi/abs/10.1145/3394171.3413876)\]
\[code\]

<p id="us-2"></p>

### 2. Learning without haze-free images

* Deep-DCP: Unsupervised single image dehazing using dark channel prior loss.
\[[paper](https://ieeexplore.ieee.org/abstract/document/8897130)\]
\[[code](https://github.com/Muhammad-Dah/Unsupervised-Single-Image-Dehazing)\]

<p id="us-3"></p>

### 3. Unsupervised image decomposition

* Double-DIP: Unsupervised Image Decomposition via Coupled Deep-Image-Priors.
\[[paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Gandelsman_Double-DIP_Unsupervised_Image_Decomposition_via_Coupled_Deep-Image-Priors_CVPR_2019_paper.html)\]
\[[code](https://github.com/yossigandelsman/DoubleDIP)\]

<p id="us-4"></p>

### 4. Zero-Shot Image Dehazing
* ZID: Zero-Shot Image Dehazing.
\[[paper](https://ieeexplore.ieee.org/abstract/document/9170880)\]
\[[code](https://github.com/liboyun/ZID)\]

* YOLY: You Only Look Yourself: Unsupervised and Untrained Single Image Dehazing Neural Network.
\[[paper](https://linkspringer.53yu.com/article/10.1007/s11263-021-01431-5)\]
\[[code](https://github.com/XLearning-SCU/2021-IJCV-YOLY)\]


<!--################################################################-->
<p id="dataset"></p>

## Hazy Dataset

Here are the commonly used datasets for dehazing task.

* D-HAZY: A dataset to evaluate quantitatively dehazing algorithms.
\[[paper](https://ieeexplore.ieee.org/abstract/document/7532754)\]
\[[code](http://cs.nyu.edu/∼silberman/datasets/nyu_depth_v2.html)\]

* HazeRD: An outdoor scene dataset and benchmark for single image dehazing.
\[[paper](https://ieeexplore.ieee.org/abstract/document/8296874)\]
\[[code](https://labsites.rochester.edu/gsharma/research/computer-vision/hazerd/)\]

* I-HAZE: A Dehazing Benchmark with Real Hazy and Haze-Free Indoor Images.
\[[paper](https://linkspringer.53yu.com/chapter/10.1007/978-3-030-01449-0_52)\]
\[[code](https://data.vision.ee.ethz.ch/cvl/ntire18//i-haze/)\]

* O-HAZE: A Dehazing Benchmark With Real Hazy and Haze-Free Outdoor Images.
\[[paper](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w13/html/Ancuti_O-HAZE_A_Dehazing_CVPR_2018_paper.html)\]
\[[code](https://data.vision.ee.ethz.ch/cvl/ntire18//o-haze/)\]

* RESIDE: Benchmarking Single-Image Dehazing and Beyond.
\[[paper](https://ieeexplore.ieee.org/abstract/document/8451944)\]
\[[code](https://sites.google.com/view/reside-dehaze-datasets)\]

* Dense-Haze: A Benchmark for Image Dehazing with Dense-Haze and Haze-Free Images.
\[[paper](https://ieeexplore.ieee.org/abstract/document/8803046)\]
\[[code](https://data.vision.ee.ethz.ch/cvl/ntire19/dense-haze/)\]

* NH-HAZE: An Image Dehazing Benchmark With Non-Homogeneous Hazy and Haze-Free Images.
\[[paper](https://openaccess.thecvf.com/content_CVPRW_2020/html/w31/Ancuti_NH-HAZE_An_Image_Dehazing_Benchmark_With_Non-Homogeneous_Hazy_and_Haze-Free_CVPRW_2020_paper.html)\]
\[[code](https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/)\]

* MRFID: End-to-End Single Image Fog Removal Using Enhanced Cycle Consistent Adversarial Networks.
\[[paper](https://ieeexplore.ieee.org/abstract/document/9139368)\]
\[[code](None)\]

* BeDDE: Dehazing Evaluation: Real-World Benchmark Datasets, Criteria, and Baselines.
\[[paper](https://ieeexplore.ieee.org/abstract/document/9099036)\]
\[[code](https://github.com/xiaofeng94/BeDDE-for-defogging)\]

* 4kDehazing: Ultra-high-definition image dehazing via multi-guided bilateral learning.
\[[paper](https://ieeexplore.ieee.org/abstract/document/9578433)\]
\[[code](https://github.com/zzr-idam/4KDehazing)\]
