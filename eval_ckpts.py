import os

ckpt_dirs = ['/roboflamingo_logs/finetune_D_multi_action_token']
ckpt_names = ["checkpoint_gripper_Temporal_hist_1_state_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_fc_768_9.pth9.pth",
              "checkpoint_gripper_Temporal_hist_1_state_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_fc_768_3.pth3.pth",
              ]

# ckpt_dirs = ['/roboflamingo_logs/finetune_D_token']
# ckpt_names = [
#               "checkpoint_gripper_Temporal_hist_1_state_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_fc_768_11.pth11.pth",
#               "checkpoint_gripper_Temporal_hist_1_state_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_fc_768_12.pth12.pth",
#               "checkpoint_gripper_Temporal_hist_1_state_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_fc_768_15.pth15.pth",
#               "checkpoint_gripper_Temporal_hist_1_state_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_fc_768_16.pth16.pth",
#               "checkpoint_gripper_Temporal_hist_1_state_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_fc_768_19.pth19.pth",
#               ]
              
for ckpt_name in ckpt_names:
    for ckpt_dir in ckpt_dirs:
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        os.system('bash robouniview/pt_eval_ckpts.bash {}'.format(ckpt_path))




