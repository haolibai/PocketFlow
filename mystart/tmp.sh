sh scripts/test.sh -n 1 \
  --learner chn-pruned-rmt \
  --data_dir_local $PF_CIFAR10_LOCAL\
  --cpr_nb_insts_reg 50\
  --cpr_prune_ratio 0.2

