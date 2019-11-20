./scripts/run_seven.sh \
  nets/resnet_at_cifar100_run.py \
  -n=2 \
  --learner uniform \
  --enbl_warm_start \
  --uql_enbl_rl_agent \
  --uql_equivalent_bits 4 \
  --save_path /opt/ml/disk/PocketFlow/full-prec-resnet20-cifar100/model.ckpt \
  --uql_save_quant_model_path /opt/ml/disk/PocketFlow/uql-resnet20-cifar100/model.ckpt \
  --batch_size 512 \
  --data_dir_local /opt/ml/disk/datasets/cifar-100-binary 
