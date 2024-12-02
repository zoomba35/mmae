# 1.Datasets





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
      --finetune /root/autodl-tmp/mae-main/ck/pre3/checkpoint-399.pth （请选择在预训练阶段生成的checkpoint）
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

