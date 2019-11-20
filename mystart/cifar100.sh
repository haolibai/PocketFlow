sh scripts/run_local.sh ./nets/resnet_at_cifar100_run.py \
  -n 1 \
  --learner uniform \
  --exec_mode train \
  --uql_weight_bits 2 \
  --uql_method vanilla \
  --save_step 2000 \
  --lrn_rate_init 1e-1 \
  --loss_w_dcy 5e-4 \
  --uql_quant_epochs 200 \
  --batch_size 256 \
  --save_path ./saver/resnet18_cifar100_fullprec/model.ckpt \
  --uql_save_quant_model_path ./saver/resnet18_cifar100_w2a32/model.ckpt \
  --data_dir_local $PF_CIFAR100_LOCAL

  # --uql_enbl_rl_agent \
  # --uql_equivalent_bits 3 \
  # --batch_size 256 \

