## [MDSupCL](https://openaccess.thecvf.com/content/WACV2024/html/Rani_Activity-Based_Early_Autism_Diagnosis_Using_a_Multi-Dataset_Supervised_Contrastive_Learning_WACV_2024_paper.html) - Activity-based Early Autism Diagnosis Using A Multi-Dataset Supervised Contrastive Learning Approach [WACV 2024](https://wacv2024.thecvf.com/)

Link to pretrained models : 
We will released model checkpoints for all the methods here. Code and pretrained model for remaining compared methods is available [here](https://github.com/asharani97/CLRE_autism).
|Model Name and pretrained model|  Hand Gesture Dataset (%)  |
|-------   | ---------------------------|
|[MSupCL]  |            84.49           |
|[MSSCL]   |            57.22           |

|Model Name and pretrained model |        Autism Dataset (%)    |
|-------                         | ---------------------------  |
|[MSupCL]                        |            99.07             |
|[MSSCL]                         |            96.95             |

## Environment Setup
Our models are trained with GPU.

```
pip install -r requirements.txt
```
## Pre-training
To pretrain our models on multi-dataset (Hand Gesture and Autism Dataset)

python3 main.py --epoch 200

## Evaluation
To evaluate results , we perform it on a specific dataset at once.

python3 eval.py --setup dataset_name  \
--weights-path /path to the checkpoint  \
--epoch 200

##Cite
```
@InProceedings{Rani_2024_WACV,
    author    = {Rani, Asha and Verma, Yashaswi},
    title     = {Activity-Based Early Autism Diagnosis Using a Multi-Dataset Supervised Contrastive Learning Approach},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {7788-7797}
}
```
