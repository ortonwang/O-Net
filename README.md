#O-Net
The codes for the work "O-Net: 

## 1. Download pre-trained swin transformer model (Swin-T)
* [Get pre-trained model in this link] (https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY?usp=sharing): Put pretrained Swin-T into folder "pretrained_ckpt/" and create dir 'chechpoint','test_log' in the root path.
* 

## 2. Prepare data

- The datasets we used are provided by TransUnet's authors. Please go to ["./datasets/README.md"](datasets/README.md) for details, or please send an Email to 1072739254@qq.com or jienengchen01 AT gmail.com to request the preprocessed data. If you would like to use the preprocessed data, please use it for research purposes and do not redistribute it (following the TransUnet's License).
You can also go to https://challenge.isic-archive.com/data/#2017 to acquire the ISIC2017 dataset. Process the label from the csv file for training. Change the imgs_train_path, imgs_val_path, imgs_test_path in the train_class_after_segmentation to the path of the corresponding path.

## 3. Environment

- Please prepare an environment with python=3.8, and then use the command "pip install -r requirements.txt" for the dependencies.

## 4. Train/Test

- Run the train script on synapse dataset. The batch size we used is 24. If you do not have enough GPU memory, the bacth size can be reduced to 12 or 6 to save memory. For more information, contact 1072739254@qq.com。

- Train

```bash
sh train.sh 
```
- Test 

```bash
sh test.sh 
```

About classification:
bash train_classification.sh

## Paper
https://doi.org/10.3389/fnins.2022.876065

## References
* [TransUnet](https://github.com/Beckschen/TransUNet)
* [SwinTransformer](https://github.com/microsoft/Swin-Transformer)
* [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet.)
* 
## Citation
@article{wangnet,
  title={O-Net: A novel framework with deep fusion of CNN and Transformer for Simultaneous Segmentation and Classification},
  author={Wang, Tao and Lan, Junlin and Han, Zixin and Hu, Ziwei and Huang, Yuxiu and Deng, Yanglin and Zhang, Hejun and Wang, Jianchao and Chen, Musheng and Jiang, Haiyan and others},
  journal={Frontiers in Neuroscience},
  pages={772},
  publisher={Frontiers}
}


# O-Net

