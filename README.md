# AREA: Attribute Extraction and Aggregation for CLIP-Based Class-Incremental Learning

<div align="center">
    <div>
        <a href='https://www.lamda.nju.edu.cn/wenzh/' target='_blank'>Zhen-Hao Xie</a>&emsp;
        <a href='https://hhdnp.github.io/' target='_blank'>Yu-Cheng Shi</a>&emsp;
        <a href='http://www.lamda.nju.edu.cn/zhoudw' target='_blank'>Da-Wei Zhou</a>&emsp;
    </div>
    <div>
    School of Artificial Intelligence, Nanjing University
    </div>
    <div>
    State Key Laboratory for Novel Software Technology, Nanjing University
    </div>
</div>

<div align="center">

  <a href="http://arxiv.org/abs/2605.28809">
    <img src="https://img.shields.io/badge/Paper-red" alt="arXiv">
  </a>

</div>

The code repository for "[AREA: Attribute Extraction and Aggregation for CLIP-Based Class-Incremental Learning](http://arxiv.org/abs/2605.28809)" in PyTorch.  If you use any content of this repo for your work, please cite the following bib entry: 

```bibtex
@inproceedings{xie2026area,
  title={AREA: Attribute Extraction and Aggregation for CLIP-Based Class-Incremental Learning},
  author={Xie, Zhen-Hao and Shi, Yu-Cheng and Zhou, Da-Wei},
  booktitle={ICML},
  year={2026}
}
```






# 📝 Introduction
Class-Incremental Learning (CIL) is important in building real-world learning systems. In CLIP-based CIL, the model performs classification by comparing similarity between visual and textual embeddings obtained from template prompts, e.g., ``a photo of a [CLASS]''. This seemingly monolithic matching process can be decomposed into two conceptually distinct stages: attribute extraction and attribute aggregation. For example, a model may recognize cat using attributes such as fur texture and whiskers. When learning a new class like car, the model must extract additional attributes like wheels and adjust how they are aggregated in the shared representation space. However, since only data from the current task is available, incremental updates can bias both attribute extraction and aggregation toward new classes, leading to catastrophic forgetting. Therefore, we propose AREA for attribute extraction and aggregation for CLIP-based CIL. To stabilize extraction, we anchor class-level visual and textual attributes on the hyperspherical embedding space via principal geodesic analysis. To stabilize aggregation, we learn lightweight task-specific experts with scoring and residual refinement, regularized by a variational information bottleneck objective. During inference, we perform routing over task attribute manifolds via optimal transport for more concise prediction. Experiments on multiple benchmarks show that AREA consistently outperforms SOTA methods.


## 🔧 Requirements

**Environment**

1 [torch 1.13.0](https://github.com/pytorch/pytorch)

2 [torchvision 0.15.0](https://github.com/pytorch/vision)

3 [open-clip 2.30.0](https://github.com/mlfoundations/open_clip/releases/tag/v2.30.0)

**Dataset**

We provide the processed datasets as follows:

- **CIFAR100**: will be automatically downloaded by the code.
- **CUB200**: Google Drive: [link](https://drive.google.com/file/d/1XbUpnWpJPnItt5zQ6sHJnsjPncnNLvWb/view?usp=sharing) or OneDrive [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EVV4pT9VJ9pBrVs2x0lcwd0BlVQCtSrdbLVfhuajMry-lA?e=L6Wjsc)
- **ImageNet-R**: Google Drive: [link](https://drive.google.com/file/d/1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EU4jyLL29CtBsZkB6y-JSbgBzWF5YHhBAUz1Qw8qM2954A?e=hlWpNW)
- **ObjectNet**: Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EZFv9uaaO1hBj7Y40KoCvYkBnuUZHnHnjMda6obiDpiIWw?e=4n8Kpy) You can also refer to the [filelist](https://drive.google.com/file/d/147Mta-HcENF6IhZ8dvPnZ93Romcie7T6/view?usp=sharing) and processing [code](https://github.com/zhoudw-zdw/RevisitingCIL/issues/2#issuecomment-2280462493) if the file is too large to download.
- **Cars**: Google Drive: [link](https://drive.google.com/file/d/1D8ReAuOPenWi6SMNUrOZhbm6ViyhDHbL/view?usp=sharing  ) or OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/g/personal/ky2409911_365_nju_edu_cn/EbT1XAstg51Mpy82uHM0D2EBJLrtzmr_V64jeBRjqyyTnQ?e=h6g1rM)
- **UCF**: Google Drive: [link](https://drive.google.com/file/d/1Ng4w310_VDqpKbc7eYaumXTOiDxI02Wc/view?usp=sharing) or OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/g/personal/ky2409911_365_nju_edu_cn/EU2qHQXjASdLh1jIl6ihZmcB6G2KvqmSw-sTlZKDE6xPbg?e=7ezvTr)
- **Aircraft**: Google Drive: [link](https://drive.google.com/file/d/1xI5r1fU0d6Nff51HuOo5w-e4sGEP46Z2/view?usp=drive_link) or OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/g/personal/ky2409911_365_nju_edu_cn/ETVliZnmPY9AvZZgcFFJ6jMB2c7TRvcq7-gso2Aqvdl_VQ?e=pWXqdP)
- **Food**: Google Drive: [link](https://drive.google.com/file/d/1rupzXpwrbxki4l-RVmsRawhz1Cm0lDY5/view?usp=drive_link) or OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/g/personal/ky2409911_365_nju_edu_cn/Eb4xfptD4L5Egus-SiYxrIcBDH1VewLGp4kzyACGF_Na_w?e=duA3Ia)
- **SUN**: OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/g/personal/ky2409911_365_nju_edu_cn/EcQq1-1pFulKstYtdknB4O8BGo0hnlDRarAwB4wFEgkx0Q?e=YZ0xYV)

These subsets are sampled from the original datasets. Please note that I do not have the right to distribute these datasets. If the distribution violates the license, I shall provide the filenames instead.

You need to modify the path of the datasets in `./utils/data.py` according to your own path. 

## 💡 Running scripts

To prepare your JSON files, refer to the settings in the `exps` folder and run the following command. All main experiments from the paper are already provided in the `exps` folder, you can simply execute them to reproduce the results found in the `logs` folder.

```
python main.py --config ./exps/area/[configname].json
```

## 🌄 About Descriptions
To keep the repository lightweight, we have only included a subset of the annotated corpus for each dataset. If you need to conduct larger-scale experiments, we also provide a script under `/descriptions` to generate the descriptive corpus. You can use this script to generate the full set of annotations.

## 🎈 Acknowledgement

This repo is based on [CIL_Survey](https://github.com/zhoudw-zdw/CIL_Survey), [PyCIL](https://github.com/G-U-N/PyCIL) and [C3Box](https://github.com/LAMDA-CL/C3Box). 


