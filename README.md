# A Multi-Masked Autoencoder with Positional Offset Encoding for Enhanced Graphical Image Reconstruction

We propose a multi-masked autoencoder (M-MAE) that employs a combination of patch-average-based masking and random masking. Furthermore, we introduce a scaling layer named LScale to improve training dynamics and incorporate a position encoding offset method to generate more efficient position codes. 

![main](https://github.com/user-attachments/assets/77149d42-7b59-44a2-9ad3-16e53f1eea7a)

# Visualization
![pngmain](https://github.com/user-attachments/assets/12398e1c-8f70-4975-832c-58abe3898b4c)

# 1.Datasets

### ImageNet-1K

> Download ImageNet-1K dataset from: https://www.image-net.org/

## Places365

> Download Places365 dataset from http://places2.csail.mit.edu/download.html

## ADE20K

> Download ADE20K dataset from https://ade20k.csail.mit.edu/

# 2.Pre-training and Fine-tuning
## Pre-training 
The pre-training instruction:

      python submitit_pretrain.py 
      --job_dir /root/autodl-tmp/mae-main 
      --nodes 1 
      --use_volta32 
      --batch_size 100 
      --model mae_vit_base_patch16 
      --norm_pix_loss 
      --mask_ratio 0.75 
      --epochs 400 
      --warmup_epochs 40 
      --blr 1.5e-4 
      --weight_decay 0.05 
      --data_path **

## Fine-tuning 
The finetune instruction:

      python submitit_finetune.py 
      --job_dir /root/autodl-tmp/mae-main 
      --nodes 1 
      --batch_size 64 
      --model vit_base_patch16 
      --finetune /root/autodl-tmp/checkpoint-399.pth
      --epochs 100 
      --blr 5e-4 
      --layer_decay 0.65 
      --weight_decay 0.05 
      --drop_path 0.1 
      --reprob 0.25 
      --mixup 0.8 
      --cutmix 1.0 
      --dist_eval 
      --data_path **

# 3. Evaluation

>Experimental results on ImageNet-1k

|Model          |Backbone       |Param.        |Epoch          |Acc1           |  
| ------------- | ------------- |------------- | ------------- | ------------- |
|ViT            | ViT-B         |86            | -             | 79.9  |
|BEiT           |ViT-B          | 87           | 800           |83.2  |
|CAE            | ViT-B         | 86           | 300           | 83.6 |
|MAE            | ViT-B         | 86           | 800           | 84.4 |
|SimMIM         | ViT-B         | 86           | 800           | 83.8  |
|LocalMIM-HOG   | ViT-B         |86            | 100           | 83.3  |
|GreenMIM       | Swin-B        | 88           | 800           | 83.8  |
|DMAE           | ViT-B         |88            | 100           | 84.0       |   
|MRA            | ResNet-50     |-             |300            | 78.4    |     
|SparseMAE      | ViT-B         | 11.3(M)      | 600           | 83.2     |     
|CIM            | ViT-B         |-             | 300           |83.3 |
|CrossMAE       | ViT-B         | -            | 800           | 83.5 |
|DailyMAE       | ViT-B         | -            |  1600         | 83.3 |
|ColorMAE       | ViT-B         | 111.91       | 1600          | 83.8 |
|M-MAE          | ViT-B         |88            |800            |84.5   |   
        
>Experimental results on Places365

|Model          |Backbone       |Acc1           |  
| ------------- | ------------- | ------------- |
|MAE            | ViT-B         |57.9|
|AlexNet        |AlexNet        |53.3|
|VGG            |VGG            |55.2|
|Deep-Narrow    |ResNet-50      |55.9|
|ResNet         |ResNet         |54.7|
|AMAT           |ViT-B          |57.8|
|KAMIM          | ViT-B         |53.0|
|M-MAE          |ViT-B          |58.2|


>Experimental results on ADE20k

|Model          |Backbone       |Pre-training Epoch           |  mIoU|
| ------------- | ------------- | ------------- |------------- |
|MoCo v3      |ViT-B   |300               |47.3|
|BEiT| ViT-B    |800                | 47.1 |
|MAE| ViT-B    |1600               |48.1 |
|CAE|ViT-B    |800                | 48.8 |
|CIM|ViT-B | 300 |43.5|
|ColorMAE| ViT-B |1600 |49.3|
|MIMIC| ViT-B | 200 |46.1|
|M-MAE                | ViT-B   | 800               |49.4|


		
