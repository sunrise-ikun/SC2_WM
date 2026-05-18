export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

flag1="--exp_name release_rxr
      --run-type train
      --exp-config run_rxr/iter_train.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 1
      IL.iters 50000
      IL.lr 7e-6
      IL.log_every 500
      IL.ml_weight 1.0
      IL.sample_ratio 0.75
      IL.decay_interval 6000
      IL.load_from_ckpt True
      IL.is_requeue True
      IL.waypoint_aug True
      IL.ckpt_to_load rxr_pretrained/ckpt.iter31100.pth
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      IL.expert_policy spl
      MODEL.pretrained_path rxr_pretrained/mlm.sap_rxr/ckpts/model_step_90000.pt
      "

flag2="--exp_name release_rxr
      --run-type eval
      --exp-config run_rxr/iter_train.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 1
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING False
      EVAL.CKPT_PATH_DIR data/checkpoints/ckpt.45600.pth
      MODEL.pretrained_path rxr_pretrained/mlm.sap_rxr/ckpts/model_step_90000.pt
      IL.back_algo control
      IL.expert_policy spl
      "

flag3="--exp_name release_rxr
      --run-type inference
      --exp-config run_rxr/iter_train.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 1
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING False
      INFERENCE.CKPT_PATH data/checkpoints/ckpt.45600.pth
      INFERENCE.PREDICTIONS_FILE preds_45600.jsonl
      MODEL.pretrained_path rxr_pretrained/mlm.sap_rxr/ckpts/model_step_90000.pt
      IL.back_algo control
      IL.expert_policy spl
      "

mode=$1
case $mode in 
      train)
      echo "###### train mode ######"
      CUDA_VISIBLE_DEVICES='0' python run.py $flag1
      ;;
      eval)
      echo "###### eval mode ######"
      CUDA_VISIBLE_DEVICES='0' python run.py $flag2
      ;;
      infer)
      echo "###### infer mode ######"
      CUDA_VISIBLE_DEVICES='0' python run.py $flag3
      ;;
esac
