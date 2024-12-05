import os

def eval_one_epoch_ddp(args, dataset_path, isTraining, llm_name, **kwarg):

    data_types = args.data_eval if hasattr(args, 'data_eval') else args.data_type
    data_types = [data_types] if not isinstance(data_types, list) else data_types
    eval_log_dir_tmp = kwarg.pop('eval_log_dir')
    print(data_types)
    if isTraining:
        args.eval_loop_num = 2 # 20
        num_sequences = 96
    else:
        args.eval_loop_num = 20 # 20
        num_sequences = 1000
    for ith in range(len(data_types)):
        try:
        # if 1:
            print(f"evaluation {data_types[ith]}")
            eval_log_dir = os.path.join(eval_log_dir_tmp, data_types[ith])

            if 'keyframe' in llm_name:
                if data_types[ith] == 'calvin':
                    from .keyframe_eval_utils import eval_one_epoch_calvin_ddp as eval_one_epoch_calvin_ddp
                    print("******** calvin test 320 sequence***********")
                    eval_one_epoch_calvin_ddp(args, dataset_path=dataset_path[ith], num_sequences=num_sequences, eval_log_dir=eval_log_dir, **kwarg)
            else:
                if data_types[ith] == 'libero': 
                    from .eval_with_libero import eval_one_epoch_calvin_ddp as eval_one_epoch_libero_ddp
                    eval_one_epoch_libero_ddp(args, dataset_path=dataset_path[ith], eval_log_dir=eval_log_dir, **kwarg)
                elif data_types[ith] == 'metaworld':
                    from .eval_with_metaworld import eval_one_epoch_calvin_ddp as eval_one_epoch_metaworld_ddp
                    eval_one_epoch_metaworld_ddp(args, dataset_path=dataset_path[ith], eval_log_dir=eval_log_dir, **kwarg)
                elif data_types[ith] == 'robocasa':
                    from .eval_with_robocasa import eval_one_epoch_calvin_ddp as eval_one_epoch_robocasa_ddp
                    eval_one_epoch_robocasa_ddp(args, dataset_path=dataset_path[ith], eval_log_dir=eval_log_dir, **kwarg)
                elif data_types[ith] == 'maniskill2':
                    from .eval_with_maniskill2 import eval_one_epoch_calvin_ddp as eval_one_epoch_maniskill2_ddp
                    eval_one_epoch_maniskill2_ddp(args, dataset_path=dataset_path[ith], eval_log_dir=eval_log_dir, **kwarg)
                elif data_types[ith] == 'rlbench':
                    from .eval_with_rlbench import eval_one_epoch_calvin_ddp as eval_one_epoch_rlbench_ddp
                    eval_one_epoch_rlbench_ddp(args, dataset_path=dataset_path[ith], eval_log_dir=eval_log_dir, **kwarg)
                elif data_types[ith] == 'colosseum':
                    from .eval_with_colosseum import eval_one_epoch_calvin_ddp as eval_one_epoch_colosseum_ddp
                    eval_one_epoch_colosseum_ddp(args, dataset_path=dataset_path[ith], eval_log_dir=eval_log_dir, **kwarg)
                elif data_types[ith] == 'calvin':
                    from .eval_utils import eval_one_epoch_calvin_ddp as eval_one_epoch_calvin_ddp
                    print("******** calvin test 320 sequence***********")
                    eval_one_epoch_calvin_ddp(args, dataset_path=dataset_path[ith], num_sequences=num_sequences, eval_log_dir=eval_log_dir, **kwarg)
                elif data_types[ith] == 'robomimic':
                    from .eval_with_robomimic import eval_one_epoch_calvin_ddp as eval_one_epoch_robomimic_ddp
                    eval_one_epoch_robomimic_ddp(args, dataset_path=dataset_path[ith], eval_log_dir=eval_log_dir, **kwarg)
                elif data_types[ith] == 'chores':
                    from .eval_with_chores import eval_one_epoch_calvin_ddp as eval_one_epoch_chores_ddp
                    eval_one_epoch_chores_ddp(args, dataset_path=dataset_path[ith], eval_log_dir=eval_log_dir, **kwarg)
                else:
                    # assert False
                    print("no data_types[ith] evaluation!!")
        except Exception as e:
            print(e)
            pass
            