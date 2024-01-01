import torch
import torch.nn.functional as F
import random
import time
import math
import os, sys
from get_opt import *
from os.path import join as pjoin
import codecs as cs
from ddpm_trainer import DDPMTrainer, build_models

if __name__ == '__main__':
    mm_num_samples = 100
    mm_num_repeats = 30
    mm_num_times = 10

    diversity_times = 300
    replication_times = 1
    batch_size = 32
    opt_path = sys.argv[1]
    dataset_opt_path = opt_path

    try:
        device_id = int(sys.argv[2])
    except:
        device_id = 0
    device = torch.device('cuda:%d' % device_id if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device_id)

    # gt_loader, gt_dataset = get_dataset_motion_loader(dataset_opt_path, batch_size, device)
    # wrapper_opt = get_opt(dataset_opt_path, device)
    # eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    opt = get_opt(opt_path, device)
    opt.dim_pose = 69
    opt.max_motion_length = 64
    encoder = build_models(opt, opt.dim_pose)
    trainer = DDPMTrainer(opt, encoder)
    trainer.load(opt.model_dir+'/latest.tar')
    
    seed = 1234
    from vqmap.utils.run import set_random_seed
    set_random_seed(seed)
    B = 4
    output = trainer.generate_batch(torch.tensor([opt.max_motion_length]*B), opt.dim_pose)
    output = output.detach().cpu()
    output = output.reshape(*output.shape[:2], -1, 3).numpy()
    print("Output: ", output.shape)

    from vqmap.utils.visualize import visualize
    
    visualize([output], 64, os.path.dirname(opt_path)+f'/vis_gen_{seed}.mp4', ['Gen'])