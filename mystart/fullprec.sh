sh scripts/run_local.sh ./nets/resnet_at_cifar10_run.py \
  -n 1 \
  --learner full-prec \
  --loss_w_dcy 1e-4 \
  --batch_size 256 \
  --save_path ./saver/resnet20_cifar10_full_prec/model.ckpt \
  --saver_path_eval ./saver/resnet20_cifar10_full_prec_eval/model.ckpt \
  --data_dir_local $PF_CIFAR10_LOCAL
