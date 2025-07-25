# Retain and Enhance Modality-Specific Information for Multimodal Remote Sensing Image Land Use/Land Cover Classification

## Abstract

Multimodal remote sensing (RS) image land use/land cover (LULC) classification using optical and synthetic aperture radar (SAR) images has raised attention for recent studies. 
Current methods primarily employ multimodal fusion operations to directly explore relationships between multimodal features and obtain fused features, leading to the loss of beneficial modality-specific information problem. 
To solve this problem, this study introduces a multimodal feature decomposition and fusion (MDF) approach combined with a visual state space (VSS) block, namely MDF-VSS block. 
The MDF-VSS block emphasizes beneficial modality-specific information and perceives shared land cover information through modality-difference and modality-share features, which are then adaptively integrated to obtain discriminative fused features. 
Based on the MDF-VSS block, an MDF decoder is designed to retain beneficial multiscale modality-specific information. 
Then, a multimodal specific information enhancement (MSIE) decoder is designed to perform modality-difference feature guided auxiliary classification tasks, further enhancing modality-specific information that is expert in classification. 
Combining the MDF and MSIE decoders, a novel retain-enhance fusion network (REF-Net) is proposed to retain and enhance modality-specific information that benefits classification, thus improving the performance of multimodal RS image LULC classification. 
Extensive experimental results obtained on three public datasets demonstrate the effectiveness of the proposed REF-Net. 

## Overview

### Pipeline

<p align="center">
  <img src="figs/pipeline.jpg" alt="pipeline" width="80%">
</p>

### MDF-VSS block
<p align="center">
  <img src="figs/MDF-VSS block.jpg" alt="pipeline" width="40%">
</p>

[//]: # (## Getting Started)

[//]: # ()
[//]: # (### Confirm your code environment)

[//]: # ()
[//]: # ()
[//]: # (### Train on DFC20 dataset)

[//]: # ()
[//]: # (```bash)

[//]: # (python3 engine/train_val.py --config )

[//]: # (```)



## Citation

If this work is helpful to you, please cite it as:
```
@ARTICLE{11090158,
  author={Wei, Tianyu and Chen, He and Liu, Wenchao and Chen, Liang and Wang, Jue},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Retain and Enhance Modality-Specific Information for Multimodal Remote Sensing Image Land Use/Land Cover Classification}, 
  year={2025},
  pages={1-1},
  keywords={Optical imaging;Decoding;Optical sensors;Context modeling;Optical reflection;Land surface;Computational modeling;Adaptive optics;Visualization;Synthetic aperture radar;Multimodal;modality-specific information;land use/land cover classification;remote sensing;optical;synthetic aperture radar (SAR)},
  doi={10.1109/TGRS.2025.3591926}}

```
If you have any question, please email to me (weity@bit.edu.cn).

[//]: # (## Acknowledgment)

[//]: # ()
[//]: # (This code is based on Segmentation Models Pytorch &#40;[code]&#40;https://github.com/qubvel-org/segmentation_models.pytorch&#41;&#41; )

[//]: # (and VMamba &#40;[paper]&#40;https://arxiv.org/abs/2401.10166&#41;, [code]&#40;https://github.com/MzeroMiko/VMamba&#41;&#41;.)

[//]: # (Thanks for their excellent works.)