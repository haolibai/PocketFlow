sh scripts/run_local.sh ./nets/resnet_at_cifar10_run.py \
  -n 1 \
  --learner uniform \
  --exec_mode train \
  --uql_enbl_rl_agent \
  --uql_equivalent_bits 2.34 \
  --uql_method vanilla \
  --save_step 3000 \
  --lrn_rate_init 1e-4 \
  --uql_quant_epochs 120 \
  --batch_size 256 \
  --save_path ./models/model.ckpt \
  --uql_save_quant_model_path ./uql_quant_cifar10/model.ckpt \
  --data_dir_local $PF_CIFAR10_LOCAL


