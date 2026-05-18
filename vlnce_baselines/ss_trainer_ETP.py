import gc
import os
import sys
import random
import warnings
from collections import defaultdict, deque
from typing import Dict, List
import jsonlines
import datetime

import copy
import lmdb
import msgpack_numpy
import numpy as np
import math
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel as DDP
from copy import deepcopy
import tqdm
from gym import Space
from habitat import Config, logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.tasks.utils import cartesian_to_polar
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.env_utils import construct_envs, construct_envs_for_rl, is_slurm_batch_job
from vlnce_baselines.common.utils import extract_instruction_tokens
from vlnce_baselines.models.graph_utils import GraphMap, MAX_DIST, calculate_vp_rel_pos_fts
from vlnce_baselines.utils import reduce_loss

from .utils import get_camera_orientations12
from .utils import (
    length2mask, dir_angle_feature_with_ele,
)
from vlnce_baselines.common.utils import dis_to_con, gather_list_and_concat
from habitat_extensions.measures import NDTW, StepsTaken
from fastdtw import fastdtw

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  # noqa: F401

import torch.distributed as distr
import gzip
import json
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler
from vlnce_baselines.common.ops import pad_tensors_wgrad, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence
import cv2
from PIL import Image
import vlnce_baselines.waypoint_networks.utils as utils
from .utils import get_camera_orientations12
from .utils import (
    length2mask, dir_angle_feature_with_ele )
from vlnce_baselines.common.utils import dis_to_con, gather_list_and_concat
from vlnce_baselines.waypoint_networks.semantic_grid import SemanticGrid
from vlnce_baselines.waypoint_networks import get_img_segmentor_from_options
from vlnce_baselines.waypoint_networks.resnetUnet import ResNetUNet
import vlnce_baselines.waypoint_networks.viz_utils as viz_utils
import matplotlib.pyplot as plt


import sys
from utils_p.prompt import Prompt

from utils_p.losses import RegressionLoss, KLLoss


from PIL import Image
import imagehash

def test():
    return 0


@baseline_registry.register_trainer(name="SS-ETP")
class RLTrainer(BaseVLNCETrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.max_len = int(config.IL.max_traj_len) #  * 0.97 transfered gt path got 0.96 spl
        self.config = config
        #self.config.defrost()
        #self.config.VIDEO_OPTION = ['disk']
        #self.config.freeze()
        #---------------------------------
    
        self.warm_n = config.warm_n
        self.prompt_alpha = config.prompt_alpha
        self.neighbor = config.neighbor
        self.image_size = config.image_size
        self.prompt = Prompt(prompt_alpha=self.prompt_alpha, image_size=self.image_size).to(self.device)
        # self.memory_bank = Memory(size=config.memory_size, dimension=self.prompt.data_prompt.numel())

        self.imagine_T = config.imagine_T
        self.problistic_loss = KLLoss(alpha=0.5)
        self.action_loss = RegressionLoss(norm=2)
        
        # Dynamic threshold adjustment: maintain a fixed-size queue for KL divergence
        self.kl_queue = deque(maxlen=1000)  # N=1000
        self.dynamic_threshold_percentile = 80  # Use percentile as threshold
        self.last_kl_value = None  # Most recent KL divergence value


    def _make_dirs(self):
        if self.config.local_rank == 0:
            self._make_ckpt_dir()
            # os.makedirs(self.lmdb_features_dir, exist_ok=True)
            if self.config.EVAL.SAVE_RESULTS:
                self._make_results_dir()

    def save_checkpoint(self, iteration: int):
        torch.save(
            obj={
                "state_dict": self.policy.state_dict(),
                "optim_state": self.optimizer.state_dict(),
                "iteration": iteration,
            },
            f=os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt.{iteration}.pth"),
        )

    def _set_config(self):
        self.split = self.config.TASK_CONFIG.DATASET.SPLIT
        self.config.defrost()
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = self.split
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = self.split
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.SIMULATOR_GPU_IDS = self.config.SIMULATOR_GPU_IDS[self.config.local_rank]
        self.config.use_pbar = not is_slurm_batch_job()
        ''' if choosing image '''
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        task_config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(task_config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                setattr(task_config.SIMULATOR, camera_template, camera_config)
                task_config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
        self.config.TASK_CONFIG = task_config
        self.config.SENSORS = task_config.SIMULATOR.AGENT_0.SENSORS
        if self.config.VIDEO_OPTION:
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            shift = 0.
            orient_dict = {
                'Back': [0, math.pi + shift, 0],            # Back
                'Down': [-math.pi / 2, 0 + shift, 0],       # Down
                'Front':[0, 0 + shift, 0],                  # Front
                'Right':[0, math.pi / 2 + shift, 0],        # Right
                'Left': [0, 3 / 2 * math.pi + shift, 0],    # Left
                'Up':   [math.pi / 2, 0 + shift, 0],        # Up
            }
            sensor_uuids = []
            #H = 224
            for sensor_type in ["RGB"]:
                sensor = getattr(self.config.TASK_CONFIG.SIMULATOR, f"{sensor_type}_SENSOR")
                for camera_id, orient in orient_dict.items():
                    camera_template = f"{sensor_type}{camera_id}"
                    camera_config = deepcopy(sensor)
                    #camera_config.WIDTH = H
                    #camera_config.HEIGHT = H
                    camera_config.ORIENTATION = orient
                    camera_config.UUID = camera_template.lower()
                    camera_config.HFOV = 90 # 90 #79  
                    sensor_uuids.append(camera_config.UUID)
                    setattr(self.config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                    self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
        self.config.freeze()

        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        self.batch_size = self.config.IL.batch_size
        
        torch.cuda.set_device(self.device)
        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://',timeout=datetime.timedelta(seconds=7200000))
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()

    def _init_envs(self):
        # for DDP to load different data
        self.config.defrost()
        self.config.TASK_CONFIG.SEED = self.config.TASK_CONFIG.SEED + self.local_rank
        self.config.freeze()

        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            auto_reset_done=False
        )
        env_num = self.envs.num_envs
        dataset_len = sum(self.envs.number_of_episodes)
        logger.info(f'LOCAL RANK: {self.local_rank}, ENV NUM: {env_num}, DATASET LEN: {dataset_len}')
        observation_space = self.envs.observation_spaces[0]
        action_space = self.envs.action_spaces[0]
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        return observation_space, action_space

    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
    ):
        start_iter = 0
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        ''' initialize the waypoint predictor here '''

        n_object_classes = 27

        ## Load the pre-trained img segmentation model
        self.img_segmentor = get_img_segmentor_from_options(n_object_classes,1.0)
        self.img_segmentor = self.img_segmentor.to(self.device)

        if self.config.GPU_NUMBERS > 1:
            self.img_segmentor = DDP(self.img_segmentor,device_ids=[self.device], output_device=self.device)
        else:
            self.img_segmentor = torch.nn.DataParallel(self.img_segmentor)

        checkpoint = torch.load("pretrained/segm.pt")
        self.img_segmentor.load_state_dict(checkpoint['models']['img_segm_model'])         
        self.img_segmentor.eval()

        self.policy.net.occupancy_map_predictor = ResNetUNet(3,3,True)
        self.policy.net.semantic_map_predictor = ResNetUNet(n_object_classes+3,n_object_classes,True)
        self.policy.net.waypoint_predictor = ResNetUNet(n_object_classes+3,1,True)

        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mse_loss = torch.nn.MSELoss()

        self.noise_filter = torch.nn.Conv2d(1, 1, (7, 7), padding=(3,3)).to(self.device)
        noise_filter_weight = torch.ones(1,1,7,7).to(self.device) #/ (7.*7.)
        self.noise_filter.weight = torch.nn.Parameter(noise_filter_weight)
        self.noise_filter.eval()

        self.img_segm_size = (128,128)
        ## Build necessary info for ground-projecting the semantic segmentation
        self._xs, self._ys = torch.tensor(np.array(np.meshgrid(np.linspace(-1,1,self.img_segm_size[0]), np.linspace(1,-1,self.img_segm_size[1]))), device=self.device)
        self._xs = self._xs.reshape(1,self.img_segm_size[0],self.img_segm_size[1])
        self._ys = self._ys.reshape(1,self.img_segm_size[0],self.img_segm_size[1])
        _x, _y = torch.tensor(np.array(np.meshgrid(np.linspace(0, self.img_segm_size[0]-1, self.img_segm_size[0]), 
                                                    np.linspace(0, self.img_segm_size[1]-1, self.img_segm_size[1]))), device=self.device)
        _xy_img = torch.cat((_x.reshape(1,self.img_segm_size[0],self.img_segm_size[1]), _y.reshape(1,self.img_segm_size[0],self.img_segm_size[1])), dim=0)
        _points2D_step = _xy_img.reshape(2, -1)
        self._points2D_step = torch.transpose(_points2D_step, 0, 1) # Npoints x 2  

        self.policy.to(self.device)

        if self.config.GPU_NUMBERS > 1:
            print('Using', self.config.GPU_NUMBERS,'GPU!')
            # find_unused_parameters=False fix ddp bug
            self.policy.net = DDP(self.policy.net.to(self.device), device_ids=[self.device],
                output_device=self.device, find_unused_parameters=True, broadcast_buffers=False)
        else:
            self.policy.net = torch.nn.DataParallel(self.policy.net.to(self.device),
                device_ids=[self.device], output_device=self.device)

        # [IMPORTANT] Set trainable parameters before creating optimizer
        # Ensures optimizer only includes parameters to be trained (vln_bert)
        # Note: checkpoint not loaded yet, using default train mode settings
        self._setup_trainable_params(mode='train')
        

        ckpt_dict = self.load_checkpoint('pretrained/cwp_predictor.pth', map_location="cpu")           
        b = [key for key in ckpt_dict["state_dict"].keys()]
        for key in b:
            if 'rgb_encoder' in key:
                ckpt_dict['state_dict'].pop(key) 
        self.policy.load_state_dict(ckpt_dict["state_dict"],strict=False)

        ckpt_dict = self.load_checkpoint('pretrained/NeRF_p16_8x8.pth', map_location="cpu")
        b = [key for key in ckpt_dict["state_dict"].keys()]
        for key in b:
            if 'rgb_encoder' in key:
                ckpt_dict['state_dict'].pop(key) 
        self.policy.load_state_dict(ckpt_dict["state_dict"],strict=False)

        if load_from_ckpt:
            ckpt_dict = self.load_checkpoint(config.IL.ckpt_to_load, map_location="cpu")           
            self.policy.load_state_dict(ckpt_dict["state_dict"],strict=False)
            start_iter = ckpt_dict["iteration"]
            
            # [IMPORTANT] Reset trainable parameters after loading checkpoint
            # Because load_state_dict might change parameter states
            self._setup_trainable_params(mode='train')
            
            # [IMPORTANT] Recreate optimizer to ensure it contains correct parameters
            # Group learning rate settings (all use config.IL.lr):
            # - Frozen: embeddings, lang_encoder, img_embeddings (not in optimizer)
            # - Group 1: CrossmodalEncoder + global_sap_head
            # - Group 2: World model components
            # - Group 3: vln_memory
            # - Group 4: Other trainable parameters
            
            # ============ Group 1: CrossmodalEncoder + global_sap_head ============
            crossmodal_params = []
            crossmodal_params += list(self.policy.net.module.vln_bert.global_encoder.encoder.parameters())
            crossmodal_params += list(self.policy.net.module.vln_bert.global_sap_head.parameters())
            crossmodal_param_ids = {id(p) for p in crossmodal_params}
            
            # ============ Group 2: World Model Components ============
            world_model_params = []
            world_model_params += list(self.policy.net.module.vln_bert.global_encoder.world_model.parameters())
            world_model_params += list(self.policy.net.module.vln_bert.global_encoder.feedback_gate.parameters())
            world_model_params += list(self.policy.net.module.vln_bert.global_encoder.feedback_delta.parameters())
            world_model_params += list(self.policy.net.module.vln_bert.global_encoder.vis_predictor.parameters())
            world_model_params += list(self.policy.net.module.vln_bert.global_encoder.enhanced_sap_head.parameters())
            world_model_param_ids = {id(p) for p in world_model_params}
            
            # ============ Group 3: vln_memory ============
            vln_memory_params = list(self.policy.net.module.vln_memory.parameters())
            vln_memory_param_ids = {id(p) for p in vln_memory_params}
            
            # ============ Group 4: Other trainable parameters ============
            other_params = [
                p for p in self.policy.parameters() 
                if p.requires_grad 
                and id(p) not in crossmodal_param_ids 
                and id(p) not in world_model_param_ids 
                and id(p) not in vln_memory_param_ids
            ]
            
            param_groups = [
                {'params': crossmodal_params, 'lr': config.IL.lr, 'name': 'crossmodal'},
                {'params': world_model_params, 'lr': config.IL.lr, 'name': 'world_model'},
                {'params': vln_memory_params, 'lr': config.IL.lr, 'name': 'vln_memory'},
                {'params': other_params, 'lr': config.IL.lr, 'name': 'other'}
            ]
            
            self.optimizer = torch.optim.AdamW(param_groups)
            
            if config.IL.is_requeue:
                try:
                    self.optimizer.load_state_dict(ckpt_dict["optim_state"])
                except:
                    print("Optim_state is not loaded")
            logger.info(f"Loaded weights from checkpoint: {config.IL.ckpt_to_load}, iteration: {start_iter}")
        else:
            # Optimizer initialization needed even without load_from_ckpt
            # Group learning rate settings (all use config.IL.lr):
            # - Frozen: embeddings, lang_encoder, img_embeddings (not in optimizer)
            # - Group 1: CrossmodalEncoder + global_sap_head
            # - Group 2: World model components
            # - Group 3: vln_memory
            # - Group 4: Other trainable parameters
            
            # ============ Group 1: CrossmodalEncoder + global_sap_head ============
            crossmodal_params = []
            crossmodal_params += list(self.policy.net.module.vln_bert.global_encoder.encoder.parameters())
            crossmodal_params += list(self.policy.net.module.vln_bert.global_sap_head.parameters())
            crossmodal_param_ids = {id(p) for p in crossmodal_params}
            
            # ============ Group 2: World Model Components ============
            world_model_params = []
            world_model_params += list(self.policy.net.module.vln_bert.global_encoder.world_model.parameters())
            world_model_params += list(self.policy.net.module.vln_bert.global_encoder.feedback_gate.parameters())
            world_model_params += list(self.policy.net.module.vln_bert.global_encoder.feedback_delta.parameters())
            world_model_params += list(self.policy.net.module.vln_bert.global_encoder.vis_predictor.parameters())
            world_model_params += list(self.policy.net.module.vln_bert.global_encoder.enhanced_sap_head.parameters())
            world_model_param_ids = {id(p) for p in world_model_params}
            
            # ============ Group 3: vln_memory ============
            vln_memory_params = list(self.policy.net.module.vln_memory.parameters())
            vln_memory_param_ids = {id(p) for p in vln_memory_params}
            
            # ============ Group 4: Other trainable parameters ============
            #gmap_pos/step_embeddings, nav_token, sprel_linear, pos_encoder, pos_imagine
            other_params = [
                p for p in self.policy.parameters() 
                if p.requires_grad 
                and id(p) not in crossmodal_param_ids 
                and id(p) not in world_model_param_ids 
                and id(p) not in vln_memory_param_ids
            ]
            
            logger.info(f"[Optimizer] crossmodal (lr={config.IL.lr}): {len(crossmodal_params)} params, {sum(p.numel() for p in crossmodal_params)/1e6:.2f}M elements")
            logger.info(f"[Optimizer] world_model (lr={config.IL.lr}): {len(world_model_params)} params, {sum(p.numel() for p in world_model_params)/1e6:.2f}M elements")
            logger.info(f"[Optimizer] vln_memory (lr={config.IL.lr}): {len(vln_memory_params)} params, {sum(p.numel() for p in vln_memory_params)/1e6:.2f}M elements")
            logger.info(f"[Optimizer] other (lr={config.IL.lr}): {len(other_params)} params, {sum(p.numel() for p in other_params)/1e6:.2f}M elements")
            
            param_groups = [
                {'params': crossmodal_params, 'lr': config.IL.lr, 'name': 'crossmodal'},
                {'params': world_model_params, 'lr': config.IL.lr, 'name': 'world_model'},
                {'params': vln_memory_params, 'lr': config.IL.lr, 'name': 'vln_memory'},
                {'params': other_params, 'lr': config.IL.lr, 'name': 'other'}
            ]
            
            self.optimizer = torch.optim.AdamW(param_groups)
        
        # Initialize GradScaler for mixed precision training (needed for both train and eval with TTA)
        self.scaler = GradScaler()
            
        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params/1e6:.2f} MB. Trainable: {params_t/1e6:.2f} MB.")
        logger.info("Finished setting up policy.")

        return start_iter

    def _teacher_action(self, batch_angles, batch_distances, candidate_lengths):
        if self.config.MODEL.task_type == 'r2r':
            cand_dists_to_goal = [[] for _ in range(len(batch_angles))]
            oracle_cand_idx = []
            for j in range(len(batch_angles)):
                for k in range(len(batch_angles[j])):
                    angle_k = batch_angles[j][k]
                    forward_k = batch_distances[j][k]
                    dist_k = self.envs.call_at(j, "cand_dist_to_goal", {"angle": angle_k, "forward": forward_k})
                    cand_dists_to_goal[j].append(dist_k)
                curr_dist_to_goal = self.envs.call_at(j, "current_dist_to_goal")
                # if within target range (which def as 3.0)
                if curr_dist_to_goal < 1.5:
                    oracle_cand_idx.append(candidate_lengths[j] - 1)
                else:
                    oracle_cand_idx.append(np.argmin(cand_dists_to_goal[j]))
            return oracle_cand_idx
        elif self.config.MODEL.task_type == 'rxr':
            kargs = []
            current_episodes = self.envs.current_episodes()
            for i in range(self.envs.num_envs):
                kargs.append({
                    'ref_path':self.gt_data[str(current_episodes[i].episode_id)]['locations'],
                    'angles':batch_angles[i],
                    'distances':batch_distances[i],
                    'candidate_length':candidate_lengths[i]
                })
            oracle_cand_idx = self.envs.call(["get_cand_idx"]*self.envs.num_envs, kargs)
            return oracle_cand_idx

    def _teacher_action_new(self, batch_gmap_vp_ids, batch_no_vp_left):
        teacher_actions = []
        cur_episodes = self.envs.current_episodes()
        for i, (gmap_vp_ids, gmap, no_vp_left) in enumerate(zip(batch_gmap_vp_ids, self.gmaps, batch_no_vp_left)):
            curr_dis_to_goal = self.envs.call_at(i, "current_dist_to_goal")
            if curr_dis_to_goal < 1.5:
                teacher_actions.append(0)
            else:
                if no_vp_left:
                    teacher_actions.append(-100)
                elif self.config.IL.expert_policy == 'spl':
                    ghost_vp_pos = [(vp, random.choice(pos)) for vp, pos in gmap.ghost_real_pos.items()]
                    ghost_dis_to_goal = [
                        self.envs.call_at(i, "point_dist_to_goal", {"pos": p[1]})
                        for p in ghost_vp_pos
                    ]
                    target_ghost_vp = ghost_vp_pos[np.argmin(ghost_dis_to_goal)][0]
                    teacher_actions.append(gmap_vp_ids.index(target_ghost_vp))
                elif self.config.IL.expert_policy == 'ndtw':
                    ghost_vp_pos = [(vp, random.choice(pos)) for vp, pos in gmap.ghost_real_pos.items()]
                    target_ghost_vp = self.envs.call_at(i, "ghost_dist_to_ref", {
                        "ghost_vp_pos": ghost_vp_pos,
                        "ref_path": self.gt_data[str(cur_episodes[i].episode_id)]['locations'],
                    })
                    teacher_actions.append(gmap_vp_ids.index(target_ghost_vp))
                else:
                    raise NotImplementedError
       
        return torch.tensor(teacher_actions).cuda()



    def _vp_feature_variable(self, obs):
        batch_rgb_fts, batch_loc_fts = [], []
        batch_nav_types, batch_view_lens = [], []

        for i in range(self.envs.num_envs):
            rgb_fts, loc_fts , nav_types = [], [], []
            cand_idxes = np.zeros(12, dtype=bool)
            cand_idxes[obs['cand_img_idxes'][i]] = True
            # cand
            rgb_fts.append(obs['cand_rgb'][i])
            loc_fts.append(obs['cand_angle_fts'][i])
            nav_types += [1] * len(obs['cand_angles'][i])
            # non-cand
            rgb_fts.append(obs['pano_rgb'][i][~cand_idxes])
            loc_fts.append(obs['pano_angle_fts'][~cand_idxes])
            nav_types += [0] * (12-np.sum(cand_idxes))
            
            batch_rgb_fts.append(torch.cat(rgb_fts, dim=0))
            batch_loc_fts.append(torch.cat(loc_fts, dim=0))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_view_lens.append(len(nav_types))
        # collate
        batch_rgb_fts = pad_tensors_wgrad(batch_rgb_fts).cuda()
        batch_loc_fts = pad_tensors_wgrad(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        return {
            'rgb_fts': batch_rgb_fts, 'loc_fts': batch_loc_fts,
            'nav_types': batch_nav_types, 'view_lens': batch_view_lens,
        }


    def _nav_gmap_variable(self, cur_vp, cur_pos, cur_ori, stepk=0):
        batch_gmap_vp_ids, batch_gmap_step_ids, batch_gmap_lens = [], [], []
        batch_gmap_img_fts, batch_gmap_pos_fts = [], []
        batch_gmap_raw_fts = []  # Added: raw visual features
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []
        
        # Get NAV token from global encoder
        nav_token = self.policy.net.module.vln_bert.global_encoder.nav_token  # [1, 1, 768]

        for i, gmap in enumerate(self.gmaps):
            node_vp_ids = list(gmap.node_pos.keys())
            ghost_vp_ids = list(gmap.ghost_pos.keys())
            if len(ghost_vp_ids) == 0:
                batch_no_vp_left.append(True)
            else:
                batch_no_vp_left.append(False)

            # Insert [NAV] token after STOP: [None(STOP), 'nav'(NAV), nodes, ghosts]
            gmap_vp_ids = [None, 'nav'] + node_vp_ids + ghost_vp_ids
            gmap_step_ids = [0, stepk+1] + [gmap.node_stepId[vp] for vp in node_vp_ids] + [0]*len(ghost_vp_ids)
            gmap_visited_masks = [0, 1] + [1] * len(node_vp_ids) + [0] * len(ghost_vp_ids)

            # Get visual features: STOP(zero), NAV(nav_token), nodes, ghosts
            node_ghost_fts = [gmap.get_node_embeds(vp) for vp in node_vp_ids] + \
                             [gmap.get_node_embeds(vp) for vp in ghost_vp_ids]
            gmap_img_fts = torch.stack(
                [torch.zeros_like(node_ghost_fts[0])] + \
                [nav_token.squeeze(0).squeeze(0)] + \
                node_ghost_fts, 
                dim=0
            )
            
            # Added: collect raw visual features (for world model)
            # Same structure as gmap_img_fts: [STOP(zero), NAV(zero), nodes, ghosts]
            gmap_raw_fts = torch.stack(
                [torch.zeros_like(node_ghost_fts[0])] + \
                [torch.zeros_like(node_ghost_fts[0])] + \
                node_ghost_fts,  # Raw visual features (from gmap.get_node_embeds)
                dim=0
            )

            # Get position features
            # For NAV token: use current node's position to other nodes, and 0 to itself
            gmap_pos_fts = gmap.get_pos_fts(
                cur_vp[i], cur_pos[i], cur_ori[i], gmap_vp_ids
            )
            # NAV token's position is same as current node (index 1 in gmap_pos_fts)
            # It's already handled in get_pos_fts with vp='nav'
            # Compute pairwise distances (NAV token uses same distances as current node)
            gmap_pair_dists = np.zeros((len(gmap_vp_ids), len(gmap_vp_ids)), dtype=np.float32)
            for j in range(1, len(gmap_vp_ids)):
                for k in range(j+1, len(gmap_vp_ids)):
                    vp1 = gmap_vp_ids[j]
                    vp2 = gmap_vp_ids[k]
                    
                    # Handle NAV token: same distances as current node
                    if vp1 == 'nav':
                        # Find current node position in the list
                        if cur_vp[i] in gmap_vp_ids:
                            cur_idx = gmap_vp_ids.index(cur_vp[i])
                            if cur_idx > k:
                                dist = gmap_pair_dists[k, cur_idx]
                            elif cur_idx < k:
                                # Will be computed when j=cur_idx
                                continue
                            else:
                                dist = 0  # Distance to itself
                        else:
                            dist = 0
                    elif vp2 == 'nav':
                        # Find current node position in the list
                        if cur_vp[i] in gmap_vp_ids:
                            cur_idx = gmap_vp_ids.index(cur_vp[i])
                            if cur_idx > j:
                                dist = gmap_pair_dists[j, cur_idx]
                            elif cur_idx < j:
                                # Already computed
                                dist = gmap_pair_dists[cur_idx, j]
                            else:
                                dist = 0  # Distance to itself
                        else:
                            dist = 0
                    elif not vp1.startswith('g') and not vp2.startswith('g'):
                        dist = gmap.shortest_dist[vp1][vp2]
                    elif not vp1.startswith('g') and vp2.startswith('g'):
                        front_dis2, front_vp2 = gmap.front_to_ghost_dist(vp2)
                        dist = gmap.shortest_dist[vp1][front_vp2] + front_dis2
                    elif vp1.startswith('g') and vp2.startswith('g'):
                        front_dis1, front_vp1 = gmap.front_to_ghost_dist(vp1)
                        front_dis2, front_vp2 = gmap.front_to_ghost_dist(vp2)
                        dist = front_dis1 + gmap.shortest_dist[front_vp1][front_vp2] + front_dis2
                    else:
                        raise NotImplementedError
                    gmap_pair_dists[j, k] = gmap_pair_dists[k, j] = dist / MAX_DIST
            
            batch_gmap_vp_ids.append(gmap_vp_ids)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_lens.append(len(gmap_vp_ids))
            batch_gmap_img_fts.append(gmap_img_fts)
            batch_gmap_raw_fts.append(gmap_raw_fts)  # Added
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
        
        # collate
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_img_fts = pad_tensors_wgrad(batch_gmap_img_fts)
        batch_gmap_raw_fts = pad_tensors_wgrad(batch_gmap_raw_fts)  # Added
        batch_gmap_pos_fts = pad_tensors_wgrad(batch_gmap_pos_fts).cuda()
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()

        bs = len(cur_vp)
        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(bs, max_gmap_len, max_gmap_len).float()
        for i in range(bs):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()

        return {
            'gmap_vp_ids': batch_gmap_vp_ids, 'gmap_step_ids': batch_gmap_step_ids,
            'gmap_img_fts': batch_gmap_img_fts, 'gmap_raw_fts': batch_gmap_raw_fts,  # Added gmap_raw_fts
            'gmap_pos_fts': batch_gmap_pos_fts, 
            'gmap_masks': batch_gmap_masks, 'gmap_visited_masks': batch_gmap_visited_masks, 'gmap_pair_dists': gmap_pair_dists,
            'no_vp_left': batch_no_vp_left,
        }
    
    def _get_nav_gt_features(self, teacher_actions, gmap_vp_ids, gmap_img_fts, cur_vp):
        """
        Get target features for NAV token learning (Ground Truth)
        
        Args:
            teacher_actions: [B] - action index from teacher
            gmap_vp_ids: List[List[str]] - vp_id list for each sample
            gmap_img_fts: [B, N, 768] - visual features of graph nodes (before attention)
            cur_vp: List[str] - vp_id list of current node
            
        Returns:
            gt_features: [B, 768] - GT target features (detached)
            valid_mask: [B] - mask for valid samples (True means valid)
        """
        batch_size = teacher_actions.size(0)
        gt_features = []
        valid_mask = []
        
        for i in range(batch_size):
            gt_idx = teacher_actions[i].item()
            
            # Invalid action: no candidate points
            if gt_idx == -100:
                gt_ft = torch.zeros_like(gmap_img_fts[i, 0, :])  # dummy feature
                valid_mask.append(False)
            # Special case: if teacher chooses STOP (index 0), GT feature is current node's feature
            elif gt_idx == 0:
                # Find current node's position in gmap_vp_ids
                if cur_vp[i] in gmap_vp_ids[i]:
                    cur_node_idx = gmap_vp_ids[i].index(cur_vp[i])
                    gt_ft = gmap_img_fts[i, cur_node_idx, :]  # current node feature
                else:
                    # If current node not found (shouldn't happen), use STOP position feature
                    gt_ft = gmap_img_fts[i, 0, :]
                valid_mask.append(True)
            else:
                # Normal case: extract feature of corresponding index from gmap_img_fts
                # Note: gmap_img_fts are raw visual features before attention
                gt_ft = gmap_img_fts[i, gt_idx, :]  # [768]
                valid_mask.append(True)
            
            gt_features.append(gt_ft)
        
        gt_features = torch.stack(gt_features, dim=0)  # [B, 768]
        valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=gt_features.device)  # [B]
        
        # Detach to avoid gradient backprop to GT features
        return gt_features.detach(), valid_mask
    
    def _get_nav_negative_samples(self, teacher_actions, gmap_vp_ids, gmap_img_fts, prev_ghost_vp_ids, cur_vp):
        """
        Get negative samples for NAV token contrastive learning
        Negative sample definition:
        1. All ghost points (regardless of age)
        2. Current node (if not GT)
        
        Args:
            teacher_actions: [B] - action index from teacher
            gmap_vp_ids: List[List[str]] - vp_id list for each sample
            gmap_img_fts: [B, N, 768] - visual features of graph nodes
            prev_ghost_vp_ids: List[List[str]] - ghost vp_ids from previous step (for distinguishing new/old ghosts)
            cur_vp: List[str] - vp_id list of current node
            
        Returns:
            negative_features: [B, max_neg, 768] - negative sample features
            neg_masks: [B, max_neg] - mask for negative samples (True means valid negative sample)
            neg_vp_ids: List[List[str]] - list of negative sample vp_ids for each sample
        """
        batch_size = teacher_actions.size(0)
        batch_negatives = []
        batch_neg_masks = []
        batch_neg_vp_ids = []  # Added: record vp_id of negative samples
        
        for i in range(batch_size):
            gt_idx = teacher_actions[i].item()
             # Check if gt_idx is valid
            if gt_idx == -100 or gt_idx < 0 or gt_idx >= len(gmap_vp_ids[i]):
                # Invalid index: collect all possible negative samples
                gt_vp_id = None
            else:
                gt_vp_id = gmap_vp_ids[i][gt_idx]
            
            negatives = []
            neg_ids = []  # Added: record negative sample ID for current sample

            for j, vp in enumerate(gmap_vp_ids[i]):
                # Skip STOP(None), NAV('nav'), and GT itself
                if vp is None or vp == 'nav' or j == gt_idx:
                    continue
                
                is_negative = False
                
                # 1. If it's a ghost point (regardless of age), it's a negative sample
                if isinstance(vp, str) and vp.startswith('g'):
                    is_negative = True
                
                # 2. If it's the current node and not GT, it's a negative sample
                elif vp == cur_vp[i]:
                    is_negative = True
                
                if is_negative:
                    negatives.append(gmap_img_fts[i, j, :])
                    neg_ids.append(vp)
            
            batch_negatives.append(negatives)
            batch_neg_vp_ids.append(neg_ids)  # Record list of negative sample IDs
            # Record number of valid negative samples
            batch_neg_masks.append([True] * len(negatives))
        
        # Pad to same length
        max_neg = max(len(negs) for negs in batch_negatives) if batch_negatives else 1
        max_neg = max(max_neg, 1)  # At least 1 to avoid empty tensor
        
        padded_negatives = []
        padded_masks = []

        
        
        for negatives, masks in zip(batch_negatives, batch_neg_masks):
            if len(negatives) == 0:
                # No negative samples, pad with zero vector
                padded_neg = torch.zeros(max_neg, gmap_img_fts.size(2), device=gmap_img_fts.device)
                padded_mask = torch.zeros(max_neg, dtype=torch.bool, device=gmap_img_fts.device)
            else:
                neg_tensor = torch.stack(negatives)  # [n_neg, 768]
                # Padding
                pad_size = max_neg - len(negatives)
                if pad_size > 0:
                    padding = torch.zeros(pad_size, gmap_img_fts.size(2), device=gmap_img_fts.device)
                    padded_neg = torch.cat([neg_tensor, padding], dim=0)
                    padded_mask = torch.tensor(
                        masks + [False] * pad_size, 
                        dtype=torch.bool, 
                        device=gmap_img_fts.device
                    )
                else:
                    padded_neg = neg_tensor
                    padded_mask = torch.tensor(masks, dtype=torch.bool, device=gmap_img_fts.device)
            
            padded_negatives.append(padded_neg)
            padded_masks.append(padded_mask)
        
        negative_features = torch.stack(padded_negatives, dim=0)  # [B, max_neg, 768]
        neg_masks = torch.stack(padded_masks, dim=0)  # [B, max_neg]
        
        return negative_features.detach(), neg_masks, batch_neg_vp_ids


    def _history_variable(self, obs):
        batch_size = obs['pano_rgb'].shape[0]
        hist_rgb_fts = obs['pano_rgb'][:, 0, ...].cuda()
        hist_pano_rgb_fts = obs['pano_rgb'].cuda()
        hist_pano_ang_fts = obs['pano_angle_fts'].unsqueeze(0).expand(batch_size, -1, -1).cuda()

        return hist_rgb_fts, hist_pano_rgb_fts, hist_pano_ang_fts

    @staticmethod
    def _pause_envs(envs, batch, envs_to_pause):
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)
            
            for k, v in batch.items():
                batch[k] = v[state_index]

        return envs, batch

    def train(self):
        self._set_config()
        if self.config.MODEL.task_type == 'rxr':
            self.gt_data = {}
            for role in self.config.TASK_CONFIG.DATASET.ROLES:
                with gzip.open(
                    self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(
                        split=self.split, role=role
                    ), "rt") as f:
                    self.gt_data.update(json.load(f))
        
        if self.config.MODEL.task_type == 'r2r':
            self.gt_data = {}
            for role in self.config.TASK_CONFIG.DATASET.ROLES:
                with gzip.open(
                    self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(
                        split=self.split, role=role
                    ), "rt") as f:
                    self.gt_data.update(json.load(f))
        
        observation_space, action_space = self._init_envs()
        start_iter = self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            observation_space=observation_space,
            action_space=action_space,
        )

        total_iter = self.config.IL.iters
        log_every  = self.config.IL.log_every
        writer     = TensorboardWriter(self.config.TENSORBOARD_DIR if self.local_rank < 1 else None)

        # self.scaler is already initialized in _initialize_policy
        logger.info('Traning Starts... GOOD LUCK!')


        for idx in range(start_iter, total_iter, log_every):
            interval = min(log_every, max(total_iter-idx, 0))
            cur_iter = idx + interval

            sample_ratio = self.config.IL.sample_ratio ** ((idx-15000) // self.config.IL.decay_interval + 1)
            # sample_ratio = self.config.IL.sample_ratio ** (idx // self.config.IL.decay_interval)
            logs = self._train_interval(interval, self.config.IL.ml_weight, sample_ratio, writer=writer, start_iter=idx)

            if self.local_rank < 1: 
                loss_str = f'iter {cur_iter}: '
                for k, v in logs.items():
                    logs[k] = np.mean(v)
                    loss_str += f'{k}: {logs[k]:.3f}, '
                    writer.add_scalar(f'loss/{k}', logs[k], cur_iter)
                logger.info(loss_str)
                self.save_checkpoint(cur_iter)



    def _save_initial_weights(self):
        """
        Save initial weights for L2 regularization in TTA.
        Saves weights of trainable parameters (global_encoder only).
        """
        self.initial_weights = {}
        
        # Save global_encoder weights
        for name, param in self.policy.net.module.vln_bert.global_encoder.named_parameters():
            full_name = f"global_encoder.{name}"
            self.initial_weights[full_name] = param.data.clone().detach()
        
        if self.local_rank < 1:
            logger.info(f"[TTA] Saved {len(self.initial_weights)} initial weight tensors (global_encoder only)")
    
    def _setup_tta_optimizer(self):
        """
        Create an independent TTA optimizer.
        This optimizer is separate from the training optimizer and starts with fresh state.
        It automatically collects all parameters with requires_grad=True set by _setup_trainable_params.
        """
        # Collect all trainable parameters (set by _setup_trainable_params)
        trainable_params = [p for p in self.policy.parameters() if p.requires_grad]
        
        if len(trainable_params) == 0:
            logger.warning("[TTA] No trainable parameters found for TTA optimizer!")
            return
        
        # Create fresh AdamW optimizer for TTA (no inherited momentum/state)
        tta_lr = getattr(self.config.IL, 'tta_lr', 1e-4)  # Default TTA learning rate
        self.tta_optimizer = torch.optim.AdamW(trainable_params, lr=tta_lr, weight_decay=0.0)
        
        ## Create fresh GradScaler for TTA
        self.tta_scaler = GradScaler()
        
        if self.local_rank < 1:
            total_params = sum(p.numel() for p in trainable_params)
            logger.info(f"[TTA] Created independent TTA optimizer")
            logger.info(f"[TTA]   - Trainable params: {len(trainable_params)} tensors, {total_params/1e6:.4f}M elements")
            logger.info(f"[TTA]   - Learning rate: {tta_lr}")
            logger.info(f"[TTA]   - Optimizer: AdamW (fresh state, no inherited momentum)")
    
    def _compute_weight_l2_loss(self, debug=False):
        """
        Compute L2 regularization loss between current weights and initial weights.
        Computes for trainable parameters (global_encoder only).
        
        Args:
            weight_decay: coefficient for L2 regularization (default: 0.01)
            debug: if True, print detailed debug information
        
        Returns:
            L2 loss as a scalar tensor
        """
        l2_loss = torch.tensor(0.0, device=self.device)
        count = 0
        max_diff_name = None
        max_diff_value = 0.0
        
        # Compute L2 loss for global_encoder
        for name, param in self.policy.net.module.vln_bert.global_encoder.named_parameters():
            full_name = f"global_encoder.{name}"
            if full_name in self.initial_weights:
                diff = param - self.initial_weights[full_name]
                param_l2 = torch.sum(diff ** 2)
                l2_loss = l2_loss + param_l2
                count += param.numel()
                
                param_l2_value = param_l2.item()
                if param_l2_value > max_diff_value:
                    max_diff_value = param_l2_value
                    max_diff_name = full_name
        
        if debug and self.local_rank < 1:
            print(f"[L2 Debug] Total trainable params (global_encoder only): {count}")
            print(f"[L2 Debug] Total L2 loss (before weight_decay): {l2_loss.item():.6f}")
            print(f"[L2 Debug] Max diff param: {max_diff_name}, L2: {max_diff_value:.6f}")
        return l2_loss
    
    def _setup_trainable_params(self, mode='train'):
        """
        Setup trainable parameters for both train and eval modes.
        
        Train mode: vln_bert and vln_memory are trainable.
        Eval mode (TTA): Only img_embeddings is trainable,
                         other modules (embeddings, lang_encoder, global_encoder, global_sap_head, vln_memory) are frozen.
        
        Args:
            mode: 'train' or 'eval', used for logging purposes
        """
        assert mode in ['train', 'eval'], f"Mode must be 'train' or 'eval', got {mode}"
        if mode == 'train':
            # Train mode: vln_bert and vln_memory in train mode (dropout enabled, BN uses batch stats)
            self.policy.eval()
            self.policy.net.module.vln_bert.train()
            self.policy.net.module.vln_memory.train()

            # 1. Freeze all parameters first
            for param in self.policy.parameters():
                param.requires_grad = False
            
            # 2. Enable all vln_bert parameters
            for param in self.policy.net.module.vln_bert.parameters():
                param.requires_grad = True
            
            # 3. Enable vln_memory parameters
            for param in self.policy.net.module.vln_memory.parameters():
                param.requires_grad = True
            
            
            # 【comment out this line when finetuning to defreeze embeddings and lang_encoder module】
            for param in self.policy.net.module.vln_bert.embeddings.parameters():
                param.requires_grad = False
            for param in self.policy.net.module.vln_bert.lang_encoder.parameters():
                param.requires_grad = False
            
            # img_embeddings is not frozen, grouped into 'other' group (lr=1e-5)
            # for param in self.policy.net.module.vln_bert.img_embeddings.parameters():
            #     param.requires_grad = False
            
            # Freeze NeRF-related modules (even though they are detached in forward, explicit freeze is safer)
            for param in self.policy.net.module.vln_bert.nerf_view_encoder.parameters():
                param.requires_grad = False
            for param in self.policy.net.module.vln_bert.rgba_mlp.parameters():
                param.requires_grad = False
            for param in self.policy.net.module.vln_bert.clip_mlp.parameters():
                param.requires_grad = False
            '''
            # Freeze global_sap_head (using enhanced_sap_head only)
            for param in self.policy.net.module.vln_bert.global_sap_head.parameters():
                param.requires_grad = False
            '''
        else:
            # Eval mode (for TTA): all modules in eval mode (dropout disabled, BN uses running stats)
            self.policy.eval()
            self.policy.net.module.vln_bert.eval()
            self.policy.net.module.vln_memory.eval()
            
            # Freeze ALL parameters first
            for param in self.policy.parameters():
                param.requires_grad = False
            
            # Enable trainable modules for TTA (global_encoder only)
            trainable_count = 0
            
            # Enable global_encoder
            for param in self.policy.net.module.vln_bert.global_encoder.parameters():
                param.requires_grad = True
                trainable_count += param.numel()
                
            # Verify rgb_encoder is always frozen
            rgb_encoder_params = list(self.policy.net.module.rgb_encoder.parameters())
            assert all(not p.requires_grad for p in rgb_encoder_params), \
                "All rgb_encoder parameters should be frozen"
            
            # Count total parameters
            total_params = sum(p.numel() for p in self.policy.parameters())
            
            if self.local_rank < 1:
                logger.info(f"[{mode.upper()}] Parameter setup: {trainable_count} trainable, "
                        f"{total_params - trainable_count} frozen")
                logger.info(f"[{mode.upper()}] TTA mode: Trainable modules (global_encoder only)")


    def _train_interval(self, interval, ml_weight, sample_ratio, writer=None, start_iter=0):
        '''
        self.policy.train()
        self.policy.net.module.vln_bert.train()
        self.policy.net.module.rgb_encoder.eval()
        self.policy.net.module.occupancy_map_predictor.eval()
        self.policy.net.module.semantic_map_predictor.eval()
        self.policy.net.module.waypoint_predictor.eval()
        self.policy.net.module.vln_bert.pos_encoder.train()
        self.policy.net.module.vln_bert.pos_imagine.train()
        self.policy.eval()
        self.policy.net.module.vln_bert.train()
        for param in self.policy.net.module.vln_bert.pos_encoder.parameters():
            param.requires_grad = True
        for param in self.policy.net.module.vln_bert.pos_imagine.parameters():
            param.requires_grad = True
        for param in self.policy.parameters():
            param.requires_grad = False
        for param in self.policy.net.module.vln_bert.parameters():
            param.requires_grad = True
        '''
        # Setup trainable parameters (only vln_bert)
        self._setup_trainable_params(mode='train')
        
        # for name, param in self.policy.named_parameters():
        #     print(name, param.requires_grad)

        if self.local_rank < 1:
            pbar = tqdm.trange(interval, leave=False, dynamic_ncols=True)
        else:
            pbar = range(interval)
        self.logs = defaultdict(list)
        self.logs['loss_cross_entropy'] = []
        
        # Save old params for update ratio calculation
        old_params = {name: param.data.clone() for name, param in self.policy.named_parameters() if param.requires_grad}

        for idx in pbar:
            global_iter = start_iter + idx + 1
            
            self.optimizer.zero_grad()
            # self.loss = 0.
            self.loss = torch.tensor(0.0, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')

            with autocast():
                self.rollout('train', ml_weight, sample_ratio, global_iter=global_iter)
            
            # Numerical stability check
            if torch.isnan(self.loss).any() or torch.isinf(self.loss).any():
                print(f"\n[ERROR] iteration {idx}: loss contains NaN or Inf, skipping update")
                continue
            
            # print(self.loss)
            self.scaler.scale(self.loss).backward() # self.loss.backward()
            # Unscale gradients before clipping to operate on true gradients
            self.scaler.unscale_(self.optimizer)
            
            # Compute module gradient norms
            if self.local_rank < 1:
                module_grad_norms = {}
                module_param_counts = {}
                
                modules_to_monitor = {
                    'vln_memory': self.policy.net.module.vln_memory,
                    'vln_bert.global_encoder': self.policy.net.module.vln_bert.global_encoder,
                    'vln_bert.img_embeddings': self.policy.net.module.vln_bert.img_embeddings,
                    'vln_bert.lang_encoder': self.policy.net.module.vln_bert.lang_encoder,
                    'vln_bert.embeddings': self.policy.net.module.vln_bert.embeddings,
                    'vln_bert.global_sap_head': self.policy.net.module.vln_bert.global_sap_head,
                }
                
                for module_name, module in modules_to_monitor.items():
                    grad_norm_sq = 0.0
                    param_count = 0
                    for p in module.parameters():
                        if p.grad is not None:
                            grad_norm_sq += p.grad.data.norm(2).item() ** 2
                        param_count += p.numel()
                    module_grad_norms[module_name] = grad_norm_sq ** 0.5
                    module_param_counts[module_name] = param_count
                
                # Total gradient norm
                total_grad_norm = sum(v ** 2 for v in module_grad_norms.values()) ** 0.5
            
            torch.nn.utils.clip_grad_norm_(parameters=self.policy.parameters(), max_norm=70, norm_type=2)
            
            self.scaler.step(self.optimizer)        # self.optimizer.step()
            self.scaler.update()
            
            # Compute Update Ratio
            if self.local_rank < 1:
                delta_norm_sq = 0.0
                theta_norm_sq = 0.0
                for name, param in self.policy.named_parameters():
                    if param.requires_grad and name in old_params:
                        delta = param.data - old_params[name]
                        delta_norm_sq += delta.norm(2).item() ** 2
                        theta_norm_sq += param.data.norm(2).item() ** 2
                        old_params[name] = param.data.clone()
                
                delta_norm = delta_norm_sq ** 0.5
                theta_norm = theta_norm_sq ** 0.5
                update_ratio = delta_norm / (theta_norm + 1e-8)
                
                # Per-iteration output
                print(f"\n[Module Grad Norms] iter {global_iter}:")
                sorted_modules = sorted(module_grad_norms.items(), key=lambda x: x[1], reverse=True)
                for module_name, grad_norm in sorted_modules:
                    param_count = module_param_counts[module_name]
                    if param_count >= 1e6:
                        param_str = f"{param_count/1e6:.2f}M"
                    else:
                        param_str = f"{param_count/1e3:.2f}K"
                    print(f"  {module_name:30s}: {grad_norm:10.4f}  (params: {param_str:>8s})")
                print(f"[Total Grad Norm] {total_grad_norm:.4f} | [Update Ratio] ||Δθ||/||θ||: {update_ratio:.6e} (||Δθ||={delta_norm:.4e}, ||θ||={theta_norm:.4e})")
            if self.local_rank < 1:
                pbar.set_postfix({'iter': f'{idx+1}/{interval}'})
            
        return deepcopy(self.logs)

    def _reset_model_to_checkpoint(self):
        """
        Reset model parameters to the original checkpoint state.
        Used in TTA to reset parameters when switching to a new scene.
        Reloads all pretrained models in the same order as _initialize_policy.
        """
        # 1. Load cwp_predictor.pth
        ckpt_dict = self.load_checkpoint('pretrained/cwp_predictor.pth', map_location="cpu")           
        b = [key for key in ckpt_dict["state_dict"].keys()]
        for key in b:
            if 'rgb_encoder' in key:
                ckpt_dict['state_dict'].pop(key) 
        self.policy.load_state_dict(ckpt_dict["state_dict"], strict=False)
        
        # 2. Load NeRF_p16_8x8.pth
        ckpt_dict = self.load_checkpoint('pretrained/NeRF_p16_8x8.pth', map_location="cpu")
        b = [key for key in ckpt_dict["state_dict"].keys()]
        for key in b:
            if 'rgb_encoder' in key:
                ckpt_dict['state_dict'].pop(key) 
        self.policy.load_state_dict(ckpt_dict["state_dict"], strict=False)
        
        # 3. Load the main checkpoint
        ckpt_dict = self.load_checkpoint(self.config.IL.ckpt_to_load, map_location="cpu")           
        self.policy.load_state_dict(ckpt_dict["state_dict"], strict=False)
        
        # Reset optimizer state
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.policy.parameters()), 
            lr=self.config.IL.lr
        )
        
        # Reset scaler
        self.scaler = GradScaler()
        
        # Re-setup trainable parameters and TTA learning rate
        self._setup_trainable_params(mode='eval')
        
        # Re-save initial weights after reset
        self._save_initial_weights()
        
        if self.local_rank < 1:
            logger.info(f"[TTA] Model parameters and optimizer fully reset to checkpoint state")
            logger.info(f"[TTA] Initial weights re-saved for L2 regularization")

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ):
        """
        Evaluate checkpoint with dynamic parameter updates enabled for vln_bert.
        Note: @torch.no_grad() decorator is removed to allow gradient computation.
        """
        if self.local_rank < 1:
            logger.info(f"checkpoint_path: {checkpoint_path}")
            logger.info("[EVAL] Dynamic parameter update enabled for vln_bert module")
        
        # Initialize scene tracking for TTA reset
        self.current_scene_id = None
        
        self.config.defrost()
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.IL.ckpt_to_load = checkpoint_path
        if self.config.VIDEO_OPTION:
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            shift = 0.
            orient_dict = {
                'Back': [0, math.pi + shift, 0],            # Back
                'Down': [-math.pi / 2, 0 + shift, 0],       # Down
                'Front':[0, 0 + shift, 0],                  # Front
                'Right':[0, math.pi / 2 + shift, 0],        # Right
                'Left': [0, 3 / 2 * math.pi + shift, 0],    # Left
                'Up':   [math.pi / 2, 0 + shift, 0],        # Up
            }
            sensor_uuids = []
            #H = 224
            for sensor_type in ["RGB"]:
                sensor = getattr(self.config.TASK_CONFIG.SIMULATOR, f"{sensor_type}_SENSOR")
                for camera_id, orient in orient_dict.items():
                    camera_template = f"{sensor_type}{camera_id}"
                    camera_config = deepcopy(sensor)
                    #camera_config.WIDTH = H
                    #camera_config.HEIGHT = H
                    camera_config.ORIENTATION = orient
                    camera_config.UUID = camera_template.lower()
                    camera_config.HFOV = 90
                    sensor_uuids.append(camera_config.UUID)
                    setattr(self.config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                    self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
        self.config.freeze()

        if self.config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                self.config.RESULTS_DIR,
                f"stats_ckpt_{checkpoint_index}_{self.config.TASK_CONFIG.DATASET.SPLIT}.json",
            )
            if os.path.exists(fname) and not os.path.isfile(self.config.EVAL.CKPT_PATH_DIR):
                print("skipping -- evaluation exists.")
                return
        self.envs = construct_envs(# Initialize simulation environments
            self.config, 
            get_env_class(self.config.ENV_NAME),
            episodes_allowed=self.traj[::3] if self.config.EVAL.fast_eval else self.traj,
            auto_reset_done=False, # unseen: 11006 
        )
        #self.traj[564:819:1]
        dataset_length = sum(self.envs.number_of_episodes)
        print('local rank:', self.local_rank, '|', 'dataset length:', dataset_length)
        
        obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            self.config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
        )
        
        # Setup trainable parameters (same as train mode: only vln_bert is trainable)
        self._setup_trainable_params(mode='eval')
        
        # Verify optimizer exists (it should be created in _initialize_policy)
        assert hasattr(self, 'optimizer'), "Optimizer should be initialized"
        assert hasattr(self, 'scaler'), "GradScaler should be initialized"
        
        # Create independent TTA optimizer (separate from training optimizer)
        self._setup_tta_optimizer()
        
        # Save initial weights for L2 regularization in TTA
        self._save_initial_weights()
        
        #self.policy.eval()  # This is now handled in _setup_trainable_params
        #self.waypoint_predictor.eval()

        if self.config.EVAL.EPISODE_COUNT == -1:
            eps_to_eval = sum(self.envs.number_of_episodes)
        else:
            eps_to_eval = min(self.config.EVAL.EPISODE_COUNT, sum(self.envs.number_of_episodes))
        self.stat_eps = {}
        self.pbar = tqdm.tqdm(total=eps_to_eval) if self.config.use_pbar else None
        
        # Initialize eval loss tracking
        self.eval_loss_history = []
        
        # ========== Eval Mode Statistics Counters Initialization (Accumulate all rollouts) ==========
        self.eval_logits_diff_large_steps = 0
        self.eval_argmax_diff_steps = 0
        self.eval_total_steps = 0
        self.eval_kl_divergences = []  # Store KL divergence for each step for distribution statistics
        
        # TTA: KL divergence dynamic queue for triggering TTA update
        from collections import deque
        self.tta_kl_queue = deque(maxlen=1000)
        
        # TTA: backward count
        self.tta_backward_count = 0
        
        while len(self.stat_eps) < eps_to_eval:
            self.rollout('eval')
        self.envs.close()

        if self.world_size > 1:
            distr.barrier()
        aggregated_states = {}
        num_episodes = len(self.stat_eps)
        for stat_key in next(iter(self.stat_eps.values())).keys():
            aggregated_states[stat_key] = (
                sum(v[stat_key] for v in self.stat_eps.values()) / num_episodes
            )
        total = torch.tensor(num_episodes).cuda()
        if self.world_size > 1:
            distr.reduce(total,dst=0)
        total = total.item()

        if self.world_size > 1:
            logger.info(f"rank {self.local_rank}'s {num_episodes}-episode results: {aggregated_states}")
            for k,v in aggregated_states.items():
                v = torch.tensor(v*num_episodes).cuda()
                cat_v = gather_list_and_concat(v,self.world_size)
                v = (sum(cat_v)/total).item()
                aggregated_states[k] = v
        
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(
            self.config.RESULTS_DIR,
            f"stats_ep_ckpt_{checkpoint_index}_{split}_r{self.local_rank}_w{self.world_size}.json",
        )
        with open(fname, "w") as f:
            json.dump(self.stat_eps, f, indent=2)

        if self.local_rank < 1:
            if self.config.EVAL.SAVE_RESULTS:
                fname = os.path.join(
                    self.config.RESULTS_DIR,
                    f"stats_ckpt_{checkpoint_index}_{split}.json",
                )
                with open(fname, "w") as f:
                    json.dump(aggregated_states, f, indent=2)

            logger.info(f"Episodes evaluated: {total}")
            
            # ========== Eval Mode Statistics Output ==========
            if self.eval_total_steps > 0:
                logger.info(f"\n===== Eval Mode Logits Diff Statistics =====")
                logger.info(f"Total steps: {self.eval_total_steps}")
                logger.info(f"Steps with stop/ghost logits diff > 2: {self.eval_logits_diff_large_steps} ({100*self.eval_logits_diff_large_steps/self.eval_total_steps:.2f}%)")
                logger.info(f"Steps with different argmax decision: {self.eval_argmax_diff_steps} ({100*self.eval_argmax_diff_steps/self.eval_total_steps:.2f}%)")
                
                # ========== KL Divergence Statistics Output ==========
                if len(self.eval_kl_divergences) > 0:
                    kl_array = np.array(self.eval_kl_divergences)
                    kl_mean = np.mean(kl_array)
                    kl_std = np.std(kl_array)
                    kl_min = np.min(kl_array)
                    kl_max = np.max(kl_array)
                    kl_median = np.median(kl_array)
                    kl_25 = np.percentile(kl_array, 25)
                    kl_75 = np.percentile(kl_array, 75)
                    kl_90 = np.percentile(kl_array, 90)
                    logger.info(f"\n----- KL Divergence Statistics (enhanced || global) -----")
                    logger.info(f"Sample count: {len(self.eval_kl_divergences)}")
                    logger.info(f"Mean: {kl_mean:.4f}, Std: {kl_std:.4f}")
                    logger.info(f"Min: {kl_min:.4f}, Max: {kl_max:.4f}")
                    logger.info(f"Median: {kl_median:.4f}")
                    logger.info(f"Percentiles: 25%={kl_25:.4f}, 75%={kl_75:.4f}, 90%={kl_90:.4f}")
                
                # ========== TTA Backward Statistics Output ==========
                logger.info(f"\n----- TTA Backward Statistics -----")
                logger.info(f"TTA Backward total count: {self.tta_backward_count}")
                logger.info(f"TTA Backward ratio: {100*self.tta_backward_count/self.eval_total_steps:.2f}%")
            
            checkpoint_num = checkpoint_index + 1
            for k, v in aggregated_states.items():
                logger.info(f"Average episode {k}: {v:.6f}")
                writer.add_scalar(f"eval_{k}/{split}", v, checkpoint_num)
            
            # Log eval loss statistics
            if len(self.eval_loss_history) > 0:
                avg_eval_loss = sum(self.eval_loss_history) / len(self.eval_loss_history)
                logger.info(f"[EVAL] Dynamic updates performed: {len(self.eval_loss_history)} times")
                logger.info(f"[EVAL] Average loss_to_calculate: {avg_eval_loss:.6f}")
                writer.add_scalar(f"eval_dynamic_loss/{split}", avg_eval_loss, checkpoint_num)
            else:
                logger.info("[EVAL] No dynamic updates were performed (all losses were zero)")

    @torch.no_grad()
    def inference(self):
        checkpoint_path = self.config.INFERENCE.CKPT_PATH
        logger.info(f"checkpoint_path: {checkpoint_path}")
        self.config.defrost()
        self.config.IL.ckpt_to_load = checkpoint_path
        self.config.TASK_CONFIG.DATASET.SPLIT = self.config.INFERENCE.SPLIT
        self.config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        self.config.TASK_CONFIG.DATASET.LANGUAGES = self.config.INFERENCE.LANGUAGES
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.TASK_CONFIG.TASK.MEASUREMENTS = ['POSITION_INFER']
        self.config.TASK_CONFIG.TASK.SENSORS = [s for s in self.config.TASK_CONFIG.TASK.SENSORS if "INSTRUCTION" in s]
        self.config.SIMULATOR_GPU_IDS = [self.config.SIMULATOR_GPU_IDS[self.config.local_rank]]
        # if choosing image
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        task_config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(task_config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                setattr(task_config.SIMULATOR, camera_template, camera_config)
                task_config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
        self.config.TASK_CONFIG = task_config
        self.config.SENSORS = task_config.SIMULATOR.AGENT_0.SENSORS
        self.config.freeze()
        
        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        torch.cuda.set_device(self.device)
        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://',timeout=datetime.timedelta(seconds=7200000))
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()

        self.traj = self.collect_infer_traj()

        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            episodes_allowed=self.traj,
            auto_reset_done=False,
        )

        obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            self.config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
        )
        self.policy.eval()
        #self.waypoint_predictor.eval()

        if self.config.INFERENCE.EPISODE_COUNT == -1:
            eps_to_infer = sum(self.envs.number_of_episodes)
        else:
            eps_to_infer = min(self.config.INFERENCE.EPISODE_COUNT, sum(self.envs.number_of_episodes))
        self.path_eps = defaultdict(list)
        self.inst_ids: Dict[str, int] = {}   # transfer submit format
        self.pbar = tqdm.tqdm(total=eps_to_infer)

        while len(self.path_eps) < eps_to_infer:
            self.rollout('infer')
        self.envs.close()

        if self.world_size > 1:
            distr.barrier()
            aggregated_path_eps = [None for _ in range(self.world_size)]
            distr.all_gather_object(aggregated_path_eps, self.path_eps)
            tmp_eps_dict = {}
            for x in aggregated_path_eps:
                tmp_eps_dict.update(x)
            self.path_eps = tmp_eps_dict

            aggregated_inst_ids = [None for _ in range(self.world_size)]
            distr.all_gather_object(aggregated_inst_ids, self.inst_ids)
            tmp_inst_dict = {}
            for x in aggregated_inst_ids:
                tmp_inst_dict.update(x)
            self.inst_ids = tmp_inst_dict


        if self.config.MODEL.task_type == "r2r":
            with open(self.config.INFERENCE.PREDICTIONS_FILE, "w") as f:
                json.dump(self.path_eps, f, indent=2)
            logger.info(f"Predictions saved to: {self.config.INFERENCE.PREDICTIONS_FILE}")
        else:  # use 'rxr' format for rxr-habitat leaderboard
            preds = []
            for k,v in self.path_eps.items():
                # save only positions that changed
                path = [v[0]["position"]]
                for p in v[1:]:
                    if p["position"] != path[-1]: path.append(p["position"])
                preds.append({"instruction_id": self.inst_ids[k], "path": path})
            preds.sort(key=lambda x: x["instruction_id"])
            with jsonlines.open(self.config.INFERENCE.PREDICTIONS_FILE, mode="w") as writer:
                writer.write_all(preds)
            logger.info(f"Predictions saved to: {self.config.INFERENCE.PREDICTIONS_FILE}")

    def get_pos_ori(self):
        pos_ori = self.envs.call(['get_pos_ori']*self.envs.num_envs)
        pos = [x[0] for x in pos_ori]
        ori = [x[1] for x in pos_ori]
        return pos, ori
    


    def renyi_entropy(self, probs, alpha):
        probs = torch.clamp(probs, min=1e-10)
        if alpha == 1:
            return -torch.sum(probs * torch.log(probs), dim=-1)
        else:
            return (1 / (alpha - 1)) * torch.log(torch.sum(probs ** alpha, dim=-1))
        
    def fuse_predictions(self, origin_p, update_p, beta, alpha):
        origin_p = torch.clamp(origin_p, min=1e-10) 
        update_p = torch.clamp(update_p, min=1e-10)
        re_original = self.renyi_entropy(probs=origin_p, alpha=alpha)
        re_updated = self.renyi_entropy(probs=update_p, alpha=alpha)
        R1 = (1 + beta) * torch.sum(re_original)
        R2 = (1 + beta) * torch.sum(re_updated)

        fused_probs = beta * origin_p / R1 + update_p / R2
        return fused_probs
    
    def find_nearest_coord_torch(self, current_coord, gt_path):
        distances = torch.norm(gt_path - current_coord, dim=1) 
        nearest_index = torch.argmin(distances).item() 
        nearest_coord = gt_path[nearest_index]  
        distance = distances[nearest_index].item() 
        return nearest_coord, nearest_index, distance
    
    def get_next_coord(self, nearest_index, gt_path_tensor):
        if nearest_index + 1 < gt_path_tensor.size(0):
            return gt_path_tensor[nearest_index + 1]
        else:
            return gt_path_tensor[nearest_index]
        


    def posref_update(self, position, pred_cur_position, ghost_pos, gmap_vp_ids, nav_logits, alpha):
        for idx, vp_id in enumerate(gmap_vp_ids[0]):
            if vp_id in ghost_pos:
                gmap_point = ghost_pos[vp_id]
                distance_ref = np.linalg.norm(pred_cur_position - gmap_point)  # L2 norm calculation
            else:
                # distance_ref = 0
                distance_ref = np.linalg.norm(pred_cur_position - position)
            weight = np.exp(-alpha * distance_ref)
            nav_logits[:, idx] += weight  # Apply the smoother weight based on distance
        return nav_logits

    

    def rollout(self, mode, ml_weight=None, sample_ratio=None, global_iter=0):


        if mode == 'train':
            feedback = 'sample'
        elif mode == 'eval' or mode == 'infer':
            feedback = 'argmax'
        else:
            raise NotImplementedError

        self.envs.resume_all()

        observations = self.envs.reset()
        instr_max_len = self.config.IL.max_text_len # r2r 80, rxr 200
        instr_pad_id = 1 if self.config.MODEL.task_type == 'rxr' else 0
        observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                                                  max_length=instr_max_len, pad_id=instr_pad_id)
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        

        if mode == 'eval':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes()) 
                            if ep.episode_id in self.stat_eps]    
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
            
            # TTA: Check if scene has changed, reset model if so
            curr_eps = self.envs.current_episodes()
            if len(curr_eps) > 0:
                # Get scene_id from first active episode
                new_scene_id = curr_eps[0].scene_id
                if self.current_scene_id is None:
                    # First episode
                    self.current_scene_id = new_scene_id
                    if self.local_rank < 1:
                        logger.info(f"[TTA] Starting with scene: {new_scene_id}")
                elif self.current_scene_id != new_scene_id:
                    # Scene changed, reset model
                    if self.local_rank < 1:
                        logger.info(f"[TTA] Scene changed: {self.current_scene_id} -> {new_scene_id}")
                        logger.info(f"[TTA] Resetting model parameters to checkpoint state")
                    self._reset_model_to_checkpoint()
                    self.current_scene_id = new_scene_id
            
            # print current loaded episode ids
            try:
                ep_ids = [ep.episode_id for ep in curr_eps]
                print(f"[eval] current episode ids: {ep_ids}")
            except Exception:
                pass
        if mode == 'infer':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes()) 
                            if ep.episode_id in self.path_eps]    
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
            curr_eps = self.envs.current_episodes()
            for i in range(self.envs.num_envs):
                if self.config.MODEL.task_type == 'rxr':
                    ep_id = curr_eps[i].episode_id
                    k = curr_eps[i].instruction.instruction_id
                    self.inst_ids[ep_id] = int(k)

        self.batch_size = self.envs.num_envs

        # encode instructions
        all_txt_ids = batch['instruction']
        all_txt_masks = (all_txt_ids != instr_pad_id)
        all_txt_embeds = self.policy.net(
            mode='language',
            txt_ids=all_txt_ids,
            txt_masks=all_txt_masks,
        )

        loss = 0.
        total_actions = 0.
        not_done_index = list(range(self.envs.num_envs))
        
        # Initialize loss counters
        count_cross_entropy = 0
        loss_cross_entropy = 0.
        
        # World model loss counters (Added)
        count_enhanced_sap = 0           # Enhanced scoring loss
        loss_enhanced_sap = 0.
        count_vis_pred = 0               # Visual prediction loss
        loss_vis_pred = 0.
        
        # World model: save previous V_pred and actual action a_t for visual prediction loss
        prev_V_pred = [None] * self.envs.num_envs
        prev_actual_action = [None] * self.envs.num_envs  # Save actual action performed in last step





        have_real_pos = (mode == 'train' or self.config.VIDEO_OPTION)
        ghost_aug = self.config.IL.ghost_aug if mode == 'train' else 0
        self.gmaps = [GraphMap(have_real_pos, 
                               self.config.IL.loc_noise, 
                               self.config.MODEL.merge_ghost,
                               ghost_aug) for _ in range(self.envs.num_envs)]
        prev_vp = [None] * self.envs.num_envs


        ##############
        loss = 0.
        total_actions = 0.

        if self.config.MODEL.task_type == 'r2r':
            hfov = 90. * np.pi / 180.
            vfov = 90. * np.pi / 180.
        elif self.config.MODEL.task_type == 'rxr':
            hfov = 79. * np.pi / 180.
            vfov = 79. * np.pi / 180.

        map_config={'hfov':hfov,'vfov':vfov,'global_dim':(512,512),'grid_dim':(192,192),'heatmap_size':192,'cell_size':0.05,'img_segm_size':(128,128),'spatial_labels':3,'object_labels':27,'img_size':[256,256],'occupancy_height_thresh':-1.0,'norm_depth':True}
        # 3d info
        xs, ys = torch.tensor(np.array(np.meshgrid(np.linspace(-1,1,map_config['img_size'][0]), np.linspace(1,-1,map_config['img_size'][1]))), device='cuda')

        xs = xs.reshape(1,map_config['img_size'][0],map_config['img_size'][1])
        ys = ys.reshape(1,map_config['img_size'][0],map_config['img_size'][1])
        K = np.array([
            [1 / np.tan(map_config['hfov'] / 2.), 0., 0., 0.],
            [0., 1 / np.tan(map_config['vfov'] / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]])
        inv_K = torch.tensor(np.linalg.inv(K), device=self.device)


        # For each episode we need a new instance of a fresh global grid
        sg_map_global = SemanticGrid(self.batch_size, map_config['global_dim'], map_config['heatmap_size'], map_config['cell_size'],
                            spatial_labels=map_config['spatial_labels'], object_labels=map_config['object_labels'])

        abs_poses = [[] for b in range(self.batch_size)]
        turn_state = [None for b in range(self.batch_size)]
        turn_observations = [None for b in range(self.batch_size)]
        positions = [None for b in range(self.batch_size)]
        headings = [None for b in range(self.batch_size)]
 
        policy_net = self.policy.net
        if hasattr(self.policy.net, 'module'):
            policy_net = self.policy.net.module

        prev_vp = [None] * self.envs.num_envs
        prev_positions_tensor = None  # <--- Added this line
        
        # TTA: store previous nav_probs and gmap_vp_ids (for test time adaptation)
        prev_nav_probs = None  # Previous navigation probability
        prev_gmap_vp_ids = None  # Previous waypoint ID list
        prev_nav_inputs = None  # Previous nav_inputs (for re-forwarding)
        prev_enhanced_logits = None  # Previous enhanced_logits (for TTA loss compute)
        
        # NAV token: store previous ghost vp_ids (to filter new ghosts as negative samples)
        prev_ghost_vp_ids = [[] for _ in range(self.envs.num_envs)]
        
        # Memory reconstruction loss: accumulator
        loss_memory_recon = torch.tensor(0.0, device=self.device)
        count_memory_recon = 0
        
        for stepk in range(self.max_len):
            batch_size = self.envs.num_envs
            # Initialize dynamic adjustment flag (reset at start of each step)
            dynamic_adjustment_triggered = False
            # agent's current position and heading
            if stepk == 0:
                num_st = 0 #new----------------------------------------------------
                
                # ⭐ Reset VLN Memory Bank (at start of each episode)
                policy_net.vln_memory.reset()
                
                for ob_i in range(batch_size):
                    agent_state_i = self.envs.call_at(ob_i,
                            "get_agent_info", {})
                    positions[ob_i] = agent_state_i['position']
                    headings[ob_i] = agent_state_i['heading']

                policy_net.start_positions = positions
                policy_net.start_headings = [(heading+2*math.pi)%(2*math.pi) for heading in headings]
                policy_net.global_fts = [[] for i in range(batch_size)]
                policy_net.global_position_x = [[] for i in range(batch_size)]
                policy_net.global_position_y = [[] for i in range(batch_size)]
                policy_net.global_position_z = [[] for i in range(batch_size)] 
                policy_net.global_patch_scales = [[] for i in range(batch_size)]
                policy_net.global_patch_directions = [[] for i in range(batch_size)]
                policy_net.global_mask = [[] for i in range(batch_size)]
            policy_net.action_step = stepk + 1
            policy_net.positions = positions
            #origin_position = copy.deepcopy(positions[0])
            policy_net.headings = [(heading+2*math.pi)%(2*math.pi) for heading in headings]

            with torch.no_grad():
                for update_id in range(2):

                    batch_img = []
                    batch_depth = []
                    batch_local3D_step = []
                    batch_rel_abs_pose = []

                    for b in range(batch_size):
                    
                        ##################################
                        if update_id == 0 and turn_observations[b]!=None:
                            # heading_vector here is agent's forward heading (global coordinates)
                            heading_vector = quaternion_rotate_vector(
                                turn_state[b].rotation.inverse(), np.array([0, 0, -1])
                            )
                            # This inverse quaternion converts a global coordinate vector to the agent's local coordinates.
                            # turn_state used for snapshots; call self.envs.call_at(b,"get_agent_state", {}) for current state

                            headings[b] = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
                            positions[b] = turn_state[b].position.tolist()
                            img = turn_observations[b]['rgb']
                            depth = turn_observations[b]['depth'].reshape(map_config['img_size'][0], map_config['img_size'][1], 1)
                            agent_state = turn_state[b]

                        else:
                            agent_state_info = self.envs.call_at(b,
                                    "get_agent_info", {})
                            positions[b] = agent_state_info['position']
                            headings[b] = agent_state_info['heading']

                            img = observations[b]['rgb']
                            depth = observations[b]['depth'].reshape(map_config['img_size'][0], map_config['img_size'][1], 1)
                            agent_state = self.envs.call_at(b,"get_agent_state", {})


    
                        ################
                        policy_net.positions[b] = positions[b] #!!!!!!!!!!!!!!!!!!!!
                        policy_net.headings[b] = headings[b]   #!!!!!!!!!!!!!!!!!!!!
                        ################

                        viz_img = img
                        #new_position = torch.tensor(positions[b])
                        img = torch.tensor(img).to(self.device)
                        
                        depth = torch.tensor(depth).to(self.device)
                        viz_depth = depth

                        if map_config['norm_depth']:
                            if self.config.MODEL.task_type == 'r2r':
                                depth_abs = utils.unnormalize_depth(depth, min=0.0, max=10.0) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                            elif self.config.MODEL.task_type == 'rxr':
                                depth_abs = utils.unnormalize_depth(depth, min=0.5, max=5.0) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                        batch_img.append(img.unsqueeze(0))
                        batch_depth.append(depth_abs.unsqueeze(0))

                        local3D_step = utils.depth_to_3D(depth_abs, map_config['img_size'], xs, ys, inv_K)
                        # local3D_step: 3D coordinates of all points in view (using depth map projection)
                        batch_local3D_step.append(local3D_step)

                        agent_pose, y_height = utils.get_sim_location(agent_state=agent_state)
                        
                        if len(abs_poses[b]) < stepk+1:
                            abs_poses[b].append(agent_pose)
                        else:
                            abs_poses[b][stepk] = agent_pose


                        # Keep track of the agent's relative pose from the initial position
                        rel_abs_pose = utils.get_rel_pose(pos2=abs_poses[b][stepk], pos1=abs_poses[b][0])
                        _rel_abs_pose = torch.Tensor(rel_abs_pose).unsqueeze(0).float()
                        _rel_abs_pose = _rel_abs_pose.to(self.device)
                        batch_rel_abs_pose.append(_rel_abs_pose)

                    new_position = torch.as_tensor(positions, dtype=torch.float32).view(-1, 3)#newnew
                    if batch_rel_abs_pose != []:
                        ### Run the img segmentation model to get the ground-projected semantic segmentation
                        batch_abs_poses = torch.tensor(abs_poses).to(self.device)
                        batch_rel_abs_pose = torch.cat(batch_rel_abs_pose,dim=0)

                        batch_img = torch.cat(batch_img,dim=0)
                        
                        batch_depth = torch.cat(batch_depth,dim=0)
                        depth_img = batch_depth.clone().permute(0,3,1,2)

                        depth_img = F.interpolate(depth_img, size=map_config['img_segm_size'], mode='nearest')
                        imgData = utils.preprocess_img(batch_img, cropSize=map_config['img_segm_size'], pixFormat='NCHW', normalize=True)

                        segm_batch = {'images':imgData.to(self.device).unsqueeze(1),
                                    'depth_imgs':depth_img.to(self.device).unsqueeze(1)}
                        
                        pred_ego_sseg, img_segm = utils.run_img_segm(model=self.img_segmentor, 
                                                                input_batch=segm_batch, 
                                                                object_labels=map_config['object_labels'], 
                                                                crop_size=map_config['global_dim'], 
                                                                cell_size=map_config['cell_size'],
                                                                xs=self._xs,
                                                                ys=self._ys,
                                                                inv_K=inv_K,
                                                                points2D_step=self._points2D_step)   

                        
                        # do ground-projection, update the projected map
                        ego_grid_sseg_3 = utils.est_occ_from_depth(batch_local3D_step, grid_dim=map_config['global_dim'], cell_size=map_config['cell_size'], 
                                                                                        device=self.device, occupancy_height_thresh=map_config['occupancy_height_thresh'])

                        # Transform the ground projected egocentric grids to geocentric using relative pose
                        occup_grid_sseg = sg_map_global.spatialTransformer(grid=ego_grid_sseg_3, pose=batch_rel_abs_pose, abs_pose=batch_abs_poses)
                        semantic_grid_sseg = sg_map_global.spatialTransformer(grid=pred_ego_sseg[:,0], pose=batch_rel_abs_pose, abs_pose=batch_abs_poses)

                        # step_geo_grid contains the map snapshot every time a new observation is added
                        global_step_occup_grid_sseg, global_step_segm_grid_sseg = sg_map_global.update_proj_grid_bayes(occup_grid_sseg.unsqueeze(1),semantic_grid_sseg.unsqueeze(1))
                    if update_id == 0 and turn_observations!=[None]*batch_size:
                        post_turn_observations = [item for item in turn_observations if item !=None]
                        post_turn_observations = extract_instruction_tokens(post_turn_observations,self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)
                        turn_batch = batch_obs(post_turn_observations, self.device)
                        turn_batch = apply_obs_transforms_batch(turn_batch, self.obs_transforms)
                        for k in turn_batch:
                            for b in range(batch_size):
                                if turn_observations[b] == None:                            
                                    turn_batch[k] = torch.cat([turn_batch[k][:b],batch[k][b:b+1],turn_batch[k][b:]],0)

                        # update the feature field
                        self.policy.net(
                            mode = "feature_field",
                            observations = turn_batch,
                            in_train = (mode == 'train' and self.config.IL.waypoint_aug),
                        )

                    elif update_id == 1:
                        self.policy.net(
                            mode = "feature_field",
                            observations = batch,
                            in_train = (mode == 'train' and self.config.IL.waypoint_aug),
                        )

                        
                #########################################################

                # transform the projected grid back to egocentric (step_ego_grid_sseg contains all preceding views at every timestep)
                step_occup_grid_sseg = sg_map_global.rotate_map(grid=global_step_occup_grid_sseg.squeeze(1), rel_pose=batch_rel_abs_pose, abs_pose=batch_abs_poses)
                step_segm_grid_sseg = sg_map_global.rotate_map(grid=global_step_segm_grid_sseg.squeeze(1), rel_pose=batch_rel_abs_pose, abs_pose=batch_abs_poses)

                # Crop the grid around the agent at each timestep
                step_occup_grid_maps = utils.crop_grid(grid=step_occup_grid_sseg, crop_size=map_config['grid_dim'])
                step_segm_grid_maps = utils.crop_grid(grid=step_segm_grid_sseg, crop_size=map_config['grid_dim'])               

                predicted_occup_grid_maps =  self.policy.net.module.occupancy_map_predictor(step_occup_grid_maps.unsqueeze(1))
                step_segm_occup_grid_maps = torch.cat((step_segm_grid_maps,predicted_occup_grid_maps),dim=-3)
                predicted_segm_grid_maps = self.policy.net.module.semantic_map_predictor(step_segm_occup_grid_maps.unsqueeze(1))
                step_segm_occup_grid_maps = torch.cat((predicted_segm_grid_maps.unsqueeze(1),predicted_occup_grid_maps.unsqueeze(1)),dim=-3)
                waypoint_grid_maps = self.policy.net.module.waypoint_predictor(step_segm_occup_grid_maps).view(batch_size,1,map_config['grid_dim'][0],map_config['grid_dim'][1]).squeeze(1)

                for b in range(batch_size):
                    waypoint_grid_maps[b] = waypoint_grid_maps[b] - waypoint_grid_maps[b].min()


                #waypoint_grid_maps = self.noise_filter(waypoint_grid_maps).squeeze(1)
                
                x = torch.arange(0, map_config['grid_dim'][0], dtype=torch.float32).to(self.device)
                y = torch.arange(0, map_config['grid_dim'][1], dtype=torch.float32).to(self.device)
                yg, xg = torch.meshgrid(y,x)
                yg = -(yg.to(self.device) -  map_config['grid_dim'][1] / 2. + 0.5)
                xg = xg.to(self.device) -  map_config['grid_dim'][0] / 2. + 0.5
                grid_rel_angle = torch.atan2(xg, yg)
                grid_rel_angle = (grid_rel_angle + 2*math.pi) % (2.*math.pi)

                predicted_waypoints = [[] for direction_idx in range(12)]

                for direction_idx in range(12):
                    back_angle = math.radians(direction_idx * 30.- 15.) 
                    front_angle = math.radians(direction_idx * 30.+ 15.)
                    if direction_idx == 0:
                        back_angle += 2.*math.pi
                        selected_part = (back_angle <= grid_rel_angle) | (grid_rel_angle <= front_angle)
                    else:
                        selected_part = (back_angle <= grid_rel_angle) & (grid_rel_angle <= front_angle)

                    tmp_waypoint_grid_maps = waypoint_grid_maps.clone()
                    tmp_waypoint_grid_maps[:,selected_part==False] = 0.
                    max_value, max_index = tmp_waypoint_grid_maps.view(batch_size,-1).max(dim=-1)
                    max_y = torch.div(max_index, map_config['grid_dim'][0], rounding_mode='floor')
                    max_x = max_index % map_config['grid_dim'][0]

                    predicted_waypoints[direction_idx] = torch.cat([max_value.view(batch_size,1),max_x.view(batch_size,1),max_y.view(batch_size,1)],dim=-1).unsqueeze(1)

                predicted_waypoints = torch.cat(predicted_waypoints,dim=1)
                
                # merge near waypoints
                merge_scale = 8
                for x_merge in range(2):
                    for y_merge in range(2):
                        tmp_predicted_waypoints = predicted_waypoints[:,:,1:].to(torch.int64)  
                        if x_merge == 1:
                            tmp_predicted_waypoints[:,:,0] = tmp_predicted_waypoints[:,:,0] + merge_scale
                        if y_merge == 1:
                            tmp_predicted_waypoints[:,:,1] = tmp_predicted_waypoints[:,:,1] + merge_scale

                        tmp_predicted_waypoints = torch.div(tmp_predicted_waypoints, merge_scale*2, rounding_mode='floor').to(torch.int32)
                        for b in range(batch_size):
                            tmp_dict = {}
                            for i in range(12):
                                # delete occupied waypoints
                                if predicted_occup_grid_maps[b,:,predicted_waypoints[b,i,1].to(torch.int64),predicted_waypoints[b,i,2].to(torch.int64)].argmax().cpu().item() == 1: # occupied
                                    predicted_waypoints[b,i,0] = 0.

                                key = str([tmp_predicted_waypoints[b][i][0].cpu().item(), tmp_predicted_waypoints[b][i][1].cpu().item()])
                                if key in tmp_dict:
                                    if predicted_waypoints[b,tmp_dict[key],0] > predicted_waypoints[b,i,0]:
                                        predicted_waypoints[b,i,0] = 0.
                                    else:
                                        predicted_waypoints[b,tmp_dict[key],0] = 0.
                                else:
                                    tmp_dict[key] = i




    
                # select k waypoints
                selected_waypoint_index = torch.topk(predicted_waypoints[:,:,0], k=8, dim=-1, largest=True)[1]
                selected_waypoints = [0 for b in range(batch_size)]
                batch_angle_idxes = []
                batch_distance_idxes = []
                for b in range(batch_size):
                    selected_waypoints[b] = predicted_waypoints[b,selected_waypoint_index[b]]
                    selected_waypoints[b] = selected_waypoints[b][selected_waypoints[b][:,0]!=0]
                    selected_waypoints[b] = selected_waypoints[b][:,1:]
                    rel_y = -(selected_waypoints[b][:,1] - map_config['grid_dim'][1]//2 + 0.5) * 0.05
                    rel_x = (selected_waypoints[b][:,0] - map_config['grid_dim'][0]//2 + 0.5) * 0.05
                    rel_angle = torch.atan2(rel_x, rel_y)

                    rel_dist = torch.sqrt(torch.square(rel_x) + torch.square(rel_y))
                    mask = (0.1 < rel_dist) & (rel_dist < 5.)
                    rel_dist = rel_dist[mask]
                    rel_angle = (rel_angle + 2*math.pi) % (2.*math.pi)
                    rel_angle = rel_angle[mask]
                    selected_waypoints[b] = selected_waypoints[b][mask]

                    # Discretization
                    angle_idx = torch.div((rel_angle+(math.pi/120)), (math.pi/60), rounding_mode='floor').to(torch.int32)
                    distance_idx = torch.div(rel_dist+0.25/2., 0.25, rounding_mode='floor').to(torch.int32) 


                    batch_angle_idxes.append(angle_idx)
                    batch_distance_idxes.append(distance_idx)
  

            # ⭐ [CRITICAL] Move waypoint call out of torch.no_grad range
            # Reason: vln_memory needs gradient backprop; must call outside no_grad
            ###############################
            total_actions += self.envs.num_envs
            txt_masks = all_txt_masks[not_done_index]
            txt_embeds = all_txt_embeds[not_done_index]

            # cand waypoint representation (requires vln_memory gradient)
            wp_outputs = self.policy.net(
                mode = "waypoint",
                batch_angle_idxes = batch_angle_idxes,
                batch_distance_idxes = batch_distance_idxes,
                observations = batch,
                in_train = (mode == 'train' and self.config.IL.waypoint_aug),
            )
            
            # ⭐ Extract Memory reconstruction loss
            if mode == 'train' and wp_outputs.get('memory_recon_loss') is not None:
                loss_memory_recon += wp_outputs['memory_recon_loss']
                count_memory_recon += 1
                # Debug output
                #print(f"[Step {stepk}] Memory Recon Loss: {wp_outputs['memory_recon_loss'].item():.4f}")

            current_positions_tensor = torch.tensor(positions, dtype=torch.float32, device=self.device)
            if stepk == 0:
                delta_pos = torch.zeros_like(current_positions_tensor)
            else:
                delta_pos = current_positions_tensor - prev_positions_tensor
            prev_positions_tensor = current_positions_tensor.clone()
            
            # pano encoder #
            vp_inputs = self._vp_feature_variable(wp_outputs)
            obser_mean = torch.mean(vp_inputs['rgb_fts'], dim=1) #---------------------------------
            vp_inputs.update({
                'mode': 'panorama',
                'delta_p': delta_pos,
            })
            pos_embedding, pano_embeds, pano_masks = self.policy.net(**vp_inputs)
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                                torch.sum(pano_masks, 1, keepdim=True) # 1*768
            
            vis_embeds = avg_pano_embeds

            combined_embeds =  torch.cat((vis_embeds, pos_embedding), dim=-1)
            # shape assertions for safety
            assert vis_embeds.dim() == 2 and vis_embeds.size(0) == self.envs.num_envs and vis_embeds.size(1) == 768, f"vis_embeds shape {vis_embeds.shape} expected [B,768]"
            assert combined_embeds.dim() == 2 and combined_embeds.size(0) == self.envs.num_envs and combined_embeds.size(1) == 1536, f"combined_embeds shape {combined_embeds.shape} expected [B,1536]"
                            

            
            
            # """   """
            # get vp_id, vp_pos of cur_node and cand_node
            cur_pos, cur_ori = self.get_pos_ori()
            cur_vp, cand_vp, cand_pos = [], [], []
            for i in range(self.envs.num_envs):
                cur_vp_i, cand_vp_i, cand_pos_i = self.gmaps[i].identify_node(
                    cur_pos[i], cur_ori[i], wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i]
                )
                cur_vp.append(cur_vp_i)
                cand_vp.append(cand_vp_i)
                cand_pos.append(cand_pos_i)
            
            if mode == 'train' or self.config.VIDEO_OPTION:
                cand_real_pos = []
                for i in range(self.envs.num_envs):
                    cand_real_pos_i = [
                        self.envs.call_at(i, "get_cand_real_pos", {"angle": ang, "forward": dis})
                        for ang, dis in zip(wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i])
                    ]
                    cand_real_pos.append(cand_real_pos_i)
            else:
                cand_real_pos = [None] * self.envs.num_envs

            for i in range(self.envs.num_envs):
                cur_embeds = avg_pano_embeds[i]
                cand_embeds = pano_embeds[i][vp_inputs['nav_types'][i]==1]
                # if mode == 'eval':
                #     cur_embeds = cur_embeds.detach()
                #     cand_embeds = cand_embeds.detach()
                self.gmaps[i].update_graph(prev_vp[i], stepk+1,
                                            cur_vp[i], cur_pos[i], cur_embeds,
                                            cand_vp[i], cand_pos[i], cand_embeds,
                                            cand_real_pos[i])

            nav_inputs = self._nav_gmap_variable(cur_vp, cur_pos, cur_ori, stepk) # cur_ori is absolute pose quaternion
            
            # TTA: save current step gmap_vp_ids for later use
            curr_gmap_vp_ids = nav_inputs['gmap_vp_ids']
            #anchor
            nav_inputs.update({
                'mode': 'navigation',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
                'stepk' : stepk,
                'enable_weight_update': False,
            })
            no_vp_left = nav_inputs.pop('no_vp_left')
            nav_outs = self.policy.net(**nav_inputs)
            nav_logits = nav_outs['global_logits']
            enhanced_logits_for_action = nav_outs.get('enhanced_logits', None)
            
            # ========== Compute mixed_nav_logits (only for selecting a_t) ==========
            # Use enhanced scoring directly (warmup logic removed)
            if enhanced_logits_for_action is not None:
                # Train/eval mode: Use enhanced scoring directly
                mixed_nav_logits = enhanced_logits_for_action
                
                if mode == 'eval':
                    
                    # ========== Eval Mode Stats: logits diff and argmax diff ==========
                    self.eval_total_steps += 1
                    
                    # Iterate through each batch to check logits diff
                    step_has_large_diff = False
                    for env_i in range(self.envs.num_envs):
                        vp_ids = nav_inputs['gmap_vp_ids'][env_i]
                        for idx in range(nav_logits.size(1)):
                            vp_id = vp_ids[idx] if idx < len(vp_ids) else None
                            # Only check stop token (idx==0) and ghost nodes (starting with 'g')
                            is_stop_or_ghost = (idx == 0) or (isinstance(vp_id, str) and vp_id.startswith('g'))
                            if is_stop_or_ghost:
                                nav_l = nav_logits[env_i, idx].item()
                                enh_l = enhanced_logits_for_action[env_i, idx].item()
                                # Only check valid logits (not -inf)
                                if nav_l > -1e6 and enh_l > -1e6:
                                    if abs(nav_l - enh_l) > 2.0:
                                        step_has_large_diff = True
                                        break
                        if step_has_large_diff:
                            break
                    
                    if step_has_large_diff:
                        self.eval_logits_diff_large_steps += 1
                    
                    # Check if argmax decisions differ
                    global_argmax = nav_logits.argmax(dim=-1)  # [B]
                    enhanced_argmax = enhanced_logits_for_action.argmax(dim=-1)  # [B]
                    if (global_argmax != enhanced_argmax).any():
                        self.eval_argmax_diff_steps += 1
                    
                    # ========== Compute KL Divergence Statistics ==========
                    # Convert logits to probability distributions (softmax over valid positions)
                    for env_i in range(self.envs.num_envs):
                        vp_ids = nav_inputs['gmap_vp_ids'][env_i]
                        num_valid = len(vp_ids)  # Number of valid nodes
                        if num_valid > 0:
                            # Extract logits for valid positions
                            global_logits_valid = nav_logits[env_i, :num_valid]
                            enhanced_logits_valid = enhanced_logits_for_action[env_i, :num_valid]
                            
                            # Convert to probability distributions
                            global_probs = F.softmax(global_logits_valid, dim=-1)
                            enhanced_probs = F.softmax(enhanced_logits_valid, dim=-1)
                            
                            # Compute KL divergence: KL(enhanced || global) = sum(enhanced * log(enhanced / global))
                            # Add epsilon to prevent log(0)
                            eps = 1e-10
                            kl_div = torch.sum(enhanced_probs * (torch.log(enhanced_probs + eps) - torch.log(global_probs + eps))).item()
                            
                            # Save KL divergence (absolute value for non-negativity)
                            self.eval_kl_divergences.append(abs(kl_div))
                            
                            # TTA: save kl_div to dynamic queue
                            self.tta_kl_queue.append(abs(kl_div))
            else:
                # If no enhanced_logits, use original nav_logits
                raise ValueError("enhanced_logits_for_action is None")
                mixed_nav_logits = nav_logits

            nav_probs1 = F.softmax(nav_logits, 1) #new
            nav_probs = nav_probs1.clone()
            mixed_nav_probs = F.softmax(mixed_nav_logits, dim=1)  # For node_stop_scores and action sampling
            
            # ========== Debug Output: nav_logits vs enhanced_logits ==========
            if stepk % 5 == 0 and mode == 'train' and enhanced_logits_for_action is not None:
                log_lines = []
                log_lines.append(f"\n[Step {stepk}] Logits Comparison (vp_id | nav_logit | enhanced_logit | enhance_prob):")
                for env_i in range(min(1, self.envs.num_envs)):  # Only print the first env
                    vp_ids = nav_inputs['gmap_vp_ids'][env_i]
                    # Compute softmax probabilities for enhanced_logits (valid nodes only)
                    num_valid = len(vp_ids)
                    enhanced_probs = F.softmax(enhanced_logits_for_action[env_i, :num_valid], dim=-1)
                    for idx in range(nav_logits.size(1)):  ## Iterate all logits positions
                        nav_l = nav_logits[env_i, idx].item()
                        enh_l = enhanced_logits_for_action[env_i, idx].item()
                        # Get enhance_prob if within valid range
                        enh_prob = enhanced_probs[idx].item() if idx < num_valid else 0.0
                        # Only print valid (non-inf) nodes
                        if nav_l > -1e6:
                            # Get vp_id, special handling for STOP and NAV
                            if idx < len(vp_ids):
                                vp_id = vp_ids[idx]
                                if vp_id is None:
                                    if idx == 0:
                                        vp_id = "[STOP]"
                                    elif idx == 1:
                                        vp_id = "[NAV]"
                                    else:
                                        vp_id = f"[idx_{idx}]"
                            else:
                                vp_id = f"[idx_{idx}]"
                            log_lines.append(f"  {str(vp_id):<12} | {nav_l:>8.4f} | {enh_l:>8.4f} | {enh_prob:>8.4f}")
                
                # Debug output disabled for release
                # for line in log_lines:
                #     print(line)
            
            # TTA: Test Time Adaptation for eval mode (based on visual prediction loss)
            if mode == 'eval':
                # Only perform TTA when stepk >= 1 (need previous step info)
                if stepk >= 1:
                    # ========== New TTA: Based on visual prediction loss ==========
                    # Calculate visual prediction loss (V_pred aligned with actual visual features at cur_vp)
                    V_pred = nav_outs.get('V_pred', None)
                    tta_vis_pred_losses = []
                    
                    # Iterate all environments to collect visual prediction losses
                    for i in range(self.envs.num_envs):
                        if prev_V_pred[i] is not None and prev_actual_action[i] is not None:
                            prev_action = prev_actual_action[i]
                            # Check if previous action was candidate node movement (index >= 2, 0=STOP, 1=NAV)
                            if prev_action >= 2 and prev_action < prev_V_pred[i].size(0):
                                # Get predicted visual features for that candidate from previous step
                                pred_visual_ft = prev_V_pred[i][prev_action]  # [768]
                                # Get actual visual features after arriving at current step
                                actual_visual_ft = pano_embeds[i, 0, :].detach()  # [768]
                                # Calculate cosine similarity loss
                                cos_sim = F.cosine_similarity(pred_visual_ft.unsqueeze(0), 
                                                              actual_visual_ft.unsqueeze(0), dim=-1)
                                tta_vis_pred_losses.append(1 - cos_sim)
                    
                    # Calculate kl_threshold_met condition
                    kl_threshold_met = False
                    if len(self.tta_kl_queue) >= 10:  # Need at least 10 samples for statistics
                        sorted_kl = sorted(self.tta_kl_queue)
                        percentile_idx = int(len(sorted_kl) * self.config.IL.tta_thres)
                        kl_percentile = sorted_kl[percentile_idx]
                        # Use most recent kl_div
                        current_kl = abs(kl_div) if 'kl_div' in dir() else 0.0
                        kl_threshold_met = current_kl > kl_percentile
                    
                    # Perform TTA update based on kl_threshold_met condition
                    if kl_threshold_met and len(tta_vis_pred_losses) > 0:
                        # Compute mean of visual prediction losses
                        tta_vis_pred_loss = torch.stack(tta_vis_pred_losses).mean()
                        
                        # Add L2 regularization term
                        weight_decay_coef = self.config.IL.tta_decay
                        l2_reg_loss_origin = self._compute_weight_l2_loss(debug=False)
                        l2_reg_loss = l2_reg_loss_origin * weight_decay_coef
                        
                        # Total loss = visual prediction loss + L2 regularization
                        total_tta_loss = tta_vis_pred_loss + l2_reg_loss
                        
                        #print(f"[TTA VisPred] step {stepk}: vis_pred_loss={tta_vis_pred_loss.item():.6f}, l2_loss(weight={weight_decay_coef:.4f})={l2_reg_loss_origin.item():.8f}")
                        
                        # Parameter update
                        self.tta_optimizer.zero_grad()
                        self.tta_scaler.scale(total_tta_loss).backward()
                        self.tta_scaler.unscale_(self.tta_optimizer)
                        
                        # Calculate and output raw gradient norm, then clip to 5.0
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            [p for group in self.tta_optimizer.param_groups for p in group['params'] if p.grad is not None],
                            max_norm=5.0
                        )
                        #print(f"[TTA Grad] step {stepk}: grad_norm_before_clip={total_norm.item():.4f}")
                        
                        self.tta_scaler.step(self.tta_optimizer)
                        self.tta_scaler.update()
                        self.tta_optimizer.zero_grad()
                        
                        # TTA backward count
                        self.tta_backward_count += 1
                        # Note: V_pred will be recomputed after constructing prev_nav_inputs later
                    else:
                        # kl_threshold not met or no valid vis_pred loss, clear gradients
                        self.tta_optimizer.zero_grad()
                        
                        # Save current V_pred for next step (use original if params not updated)
                        if V_pred is not None:
                            for i in range(self.envs.num_envs):
                                prev_V_pred[i] = V_pred[i].clone().detach()
                        

                
                if stepk >= 1:
                    # Recompute current visual features (using updated model parameters)
                    pos_embedding_prev, pano_embeds_prev, pano_masks_prev = self.policy.net(**vp_inputs)
                    avg_pano_embeds_prev = torch.sum(pano_embeds_prev * pano_masks_prev.unsqueeze(2), 1) / \
                                    torch.sum(pano_masks_prev, 1, keepdim=True) # 1*768
                
                    vis_embeds_prev = avg_pano_embeds_prev
                    combined_embeds_prev =  torch.cat((vis_embeds_prev, pos_embedding_prev), dim=-1)

                    # Update graph using updated embeddings
                    for i in range(self.envs.num_envs):
                        # Detach all existing tensors in graph before update to avoid gradient conflicts
                        # Detach node_embeds
                        for vp in list(self.gmaps[i].node_embeds.keys()):
                            if isinstance(self.gmaps[i].node_embeds[vp], torch.Tensor):
                                self.gmaps[i].node_embeds[vp] = self.gmaps[i].node_embeds[vp].detach()
                        
                        # Detach ghost_embeds (format: [tensor, count])
                        for gvp in list(self.gmaps[i].ghost_embeds.keys()):
                            if isinstance(self.gmaps[i].ghost_embeds[gvp], list) and len(self.gmaps[i].ghost_embeds[gvp]) > 0:
                                if isinstance(self.gmaps[i].ghost_embeds[gvp][0], torch.Tensor):
                                    self.gmaps[i].ghost_embeds[gvp][0] = self.gmaps[i].ghost_embeds[gvp][0].detach()
                        
                        cur_embeds_prev = avg_pano_embeds_prev[i]
                        cand_embeds_prev = pano_embeds_prev[i][vp_inputs['nav_types'][i]==1]
                        
                        # Get original embeddings used in first update_graph
                        cur_embeds_old = avg_pano_embeds[i].detach()
                        cand_embeds_old = pano_embeds[i][vp_inputs['nav_types'][i]==1].detach()
                        
                        # Replace embeddings in gmap directly instead of calling update_graph again
                        # 1. Replace current node embedding
                        self.gmaps[i].node_embeds[cur_vp[i]] = cur_embeds_prev
                        
                        # 2. Replace corresponding ghost embeddings of candidates
                        # Logic: ghost_embeds was accumulated with cand_embeds_old; replace with cand_embeds_prev
                        # i.e., ghost_embeds[gvp][0] = ghost_embeds[gvp][0] - cand_embeds_old + cand_embeds_prev
                        for cand_idx, (cvp, cpos, cembeds_prev, cembeds_old) in enumerate(zip(
                            cand_vp[i], cand_pos[i], cand_embeds_prev, cand_embeds_old)):
                            # Try to localize the node or ghost corresponding to this candidate
                            localized_nvp = self.gmaps[i]._localize(cpos, self.gmaps[i].node_pos)
                            if localized_nvp is None:  # Is a ghost node
                                localized_gvp = self.gmaps[i]._localize(cpos, self.gmaps[i].ghost_mean_pos)
                                assert localized_gvp is not None, f"localized_gvp is None for cand_idx {cand_idx}"
                                # Replace ghost node embedding: subtract old, add new (keeping count unchanged)
                                self.gmaps[i].ghost_embeds[localized_gvp][0] = \
                                    self.gmaps[i].ghost_embeds[localized_gvp][0] - cembeds_old + cembeds_prev

                    
                    # Create new prev_nav_inputs with updated graph info for next step
                    prev_nav_inputs = self._nav_gmap_variable(cur_vp, cur_pos, cur_ori, stepk)
                    prev_nav_inputs.update({
                        'mode': 'navigation',
                        'txt_embeds': txt_embeds,  # txt embeddings
                        'txt_masks': txt_masks,    # txt masks
                        'stepk': stepk,             # current step count
                        'enable_weight_update': False,
                    })
                    prev_nav_inputs.pop('no_vp_left')
                    with torch.enable_grad():
                        prev_nav_outs = self.policy.net(**prev_nav_inputs)
                        prev_enhanced_logits = prev_nav_outs['enhanced_logits']
                        # Get refreshed V_pred from prev_nav_outs to avoid gradient conflicts with old params
                        V_pred_refresh = prev_nav_outs.get('V_pred', None)
                        if V_pred_refresh is not None:
                            for i in range(self.envs.num_envs):
                                prev_V_pred[i] = V_pred_refresh[i].clone()
                    if stepk > 0:
                        nav_logits = prev_nav_outs['global_logits'].clone().detach()  # Keep nav_logits updated for other logic
                        # TTA Improvement: use updated logits for action decision
                        mixed_nav_logits = prev_enhanced_logits.clone().detach()
                        mixed_nav_probs = F.softmax(mixed_nav_logits, dim=1)



            for i, gmap in enumerate(self.gmaps):
                gmap.node_stop_scores[cur_vp[i]] = mixed_nav_probs[i, 0].data.item()
            
            """ 
            if mode == "eval":
                update_nav_logits = self.posref_update(position=new_position, pred_cur_position=pred_cur_position, 
                 ghost_pos=gmap.ghost_aug_pos, gmap_vp_ids=nav_inputs['gmap_vp_ids'], nav_logits=nav_logits.clone(), alpha=0.8)
                nav_logits = update_nav_logits.clone()
            """
            
            if mode == 'train' or self.config.VIDEO_OPTION:
                teacher_actions = self._teacher_action_new(nav_inputs['gmap_vp_ids'], no_vp_left)
            if mode == 'train':
                # Cross-entropy loss
                cross_entropy_loss_step = F.cross_entropy(nav_logits, teacher_actions, reduction='sum', ignore_index=-100)
                # Apply weight reduction if dynamic adjustment was triggered
                if dynamic_adjustment_triggered:
                    cross_entropy_loss_step = cross_entropy_loss_step * 0.2
                loss += cross_entropy_loss_step
                count_cross_entropy += 1
                
                # ========== World Model Loss ==========
                # 1. Enhanced scoring loss (enhanced_logits)
                enhanced_logits = nav_outs.get('enhanced_logits', None)
                if enhanced_logits is not None:
                    enhanced_sap_loss_step = F.cross_entropy(enhanced_logits, teacher_actions, reduction='sum', ignore_index=-100)
                    loss_enhanced_sap += enhanced_sap_loss_step
                    count_enhanced_sap += 1
                
                # 2. Visual prediction loss (V_pred aligned with next step cur_vp)
                # Use prev step's V_pred and executed action, align with current step's visual features
                V_pred = nav_outs.get('V_pred', None)
                
                if stepk > 0:
                    # Collect valid V_pred loss samples using prev step's V_pred and executed action
                    vis_pred_losses = []
                    for i in range(batch_size):
                        if prev_V_pred[i] is not None and prev_actual_action[i] is not None:
                            prev_action = prev_actual_action[i]
                            # Check if prev action was a candidate node move (index>=2, 0=STOP, 1=NAV)
                            # V_pred shape is [N, 768], first two positions (STOP, NAV) are zero placeholders
                            if prev_action >= 2 and prev_action < prev_V_pred[i].size(0):
                                pred_visual_ft = prev_V_pred[i][prev_action]  # [768]
                                # Monocular VLN: use pano_embeds index 0 (forward view)
                                actual_visual_ft = pano_embeds[i, 0, :].detach()  # [768]
                                # Cosine similarity loss
                                cos_sim = F.cosine_similarity(pred_visual_ft.unsqueeze(0), 
                                                              actual_visual_ft.unsqueeze(0), dim=-1)
                                vis_pred_losses.append(1 - cos_sim)
                    
                    if len(vis_pred_losses) > 0:
                        vis_pred_loss_step = torch.stack(vis_pred_losses).mean()
                        loss_vis_pred += vis_pred_loss_step
                        count_vis_pred += 1
                
                # Save current step's V_pred for next step (a_t saved after action selection)
                if V_pred is not None:
                    for i in range(batch_size):
                        prev_V_pred[i] = V_pred[i].clone()
                
            
            # 🔥 NaN Safety Protection: Detect NaN in eval mode and set forced STOP flag
            if mode != 'train':
                nan_detected_in_step = False
                if torch.isnan(nav_logits).any() or torch.isnan(nav_probs).any():
                    logger.info(f"[ERROR] step {stepk} (eval mode): nav_logits/nav_probs contain NaN!")
                    logger.info(f"  nav_logits has NaN: {torch.isnan(nav_logits).sum().item()} / {nav_logits.numel()}")
                    logger.info(f"  nav_probs has NaN: {torch.isnan(nav_probs).sum().item()} / {nav_probs.numel()}")
                    logger.info(f"  Forcing STOP action to avoid error propagation")
                    nan_detected_in_step = True
            
            # 🔥 If NaN detected, force STOP action (a_t=0)
            if 'nan_detected_in_step' in locals() and nan_detected_in_step:
                logger.info(f"[NaN Protection] Step {stepk}: Forcing STOP action for all environments")
                a_t = torch.zeros(self.envs.num_envs, dtype=torch.long, device=nav_logits.device)
            else:
                # Normal action selection logic
                if feedback == 'sample':
                    # Train mode: probability sampling using mixed_nav_logits
                    mixed_nav_probs = F.softmax(mixed_nav_logits, dim=1)
                    # 🔥 Critical Guard: Check nav_probs before creating Categorical distribution
                    if torch.isnan(mixed_nav_probs).any() or torch.isinf(mixed_nav_probs).any() or (mixed_nav_probs.sum(dim=1) == 0).any():
                        # Use uniform distribution as fallback
                        mixed_nav_probs = torch.ones_like(mixed_nav_probs) / mixed_nav_probs.size(1)
                        # 🔥 Re-apply visited_masks to ensure NAV and visited nodes are not selected
                        mixed_nav_probs.masked_fill_(nav_inputs['gmap_visited_masks'], 0.0)
                        mixed_nav_probs.masked_fill_(nav_inputs['gmap_masks'].logical_not(), 0.0)
                        # Re-normalization
                        mixed_nav_probs = mixed_nav_probs / mixed_nav_probs.sum(dim=1, keepdim=True).clamp(min=1e-10)
                    
                    c = torch.distributions.Categorical(mixed_nav_probs)
                    a_t = c.sample().detach()
                    a_t = torch.where(torch.rand_like(a_t, dtype=torch.float) <= sample_ratio, teacher_actions, a_t)
                elif feedback == 'argmax':
                    a_t = mixed_nav_logits.argmax(dim=-1)
                else:
                    raise NotImplementedError

            cpu_a_t = a_t.cpu().numpy()
            
            # World model: save current actual action a_t for next visual prediction loss calculation
            # Note: TTA in eval mode also needs prev_actual_action, so it's no longer restricted to mode=='train'
            for i in range(batch_size):
                prev_actual_action[i] = cpu_a_t[i]

            # make equiv action
            env_actions = []
            use_tryout = (self.config.IL.tryout and not self.config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING)
            for i, gmap in enumerate(self.gmaps):
                if cpu_a_t[i] == 0 or stepk == self.max_len - 1 or no_vp_left[i]:
                    # Detect and print cases where agent stops at the first step
                    if stepk == 0 and cpu_a_t[i] == 0:
                        curr_eps = self.envs.current_episodes()
                        ep_id = curr_eps[i].episode_id if i < len(curr_eps) else "unknown"
                        logger.info(f"[WARNING] Agent stopped at stepk=0 (first step) for episode {ep_id}, env_idx={i}")
                        #raise RuntimeError("Agent stopped at stepk=0 (first step) for episode {ep_id}, env_idx={i}")
                    
                    # stop at node with max stop_prob
                    vp_stop_scores = [(vp, stop_score) for vp, stop_score in gmap.node_stop_scores.items()]
                    stop_scores = [s[1] for s in vp_stop_scores]
                    stop_vp = vp_stop_scores[np.argmax(stop_scores)][0]
                    stop_pos = gmap.node_pos[stop_vp]
                    if self.config.IL.back_algo == 'control':
                        back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][stop_vp]]
                        back_path = back_path[1:]
                    else:
                        back_path = None

                    pred_pos = None
                    
                    vis_info = {
                            'nodes': list(gmap.node_pos.values()),
                            'ghosts': list(gmap.ghost_aug_pos.values()),
                            'predict_ghost': stop_pos,
                            'imagine_path': None,  # Add path points predicted by imagine
                            'pred_position': pred_pos,  # Add predicted current position
                    }
                    env_actions.append(
                        {
                            'action': {
                                'act': 0,
                                'cur_vp': cur_vp[i],
                                'stop_vp': stop_vp, 'stop_pos': stop_pos,
                                'back_path': back_path,
                                'tryout': use_tryout,
                            },
                            'vis_info': vis_info,
                        }
                    )
                else:
                    ghost_vp = nav_inputs['gmap_vp_ids'][i][cpu_a_t[i]]
                    # Fix KeyError: ghost_vp can be a normal node or a ghost node
                    if ghost_vp in gmap.ghost_aug_pos:
                        ghost_pos = gmap.ghost_aug_pos[ghost_vp]
                    elif ghost_vp in gmap.node_pos:
                        ghost_pos = gmap.node_pos[ghost_vp]
                    else:
                        # Should not happen if gmap_vp_ids are consistent with gmap
                        # If 'nav' is selected (index 1), it falls here. Treat as current position or handle error.
                        if ghost_vp == 'nav':
                             ghost_pos = gmap.node_pos[cur_vp[i]]
                        else:
                             logger.error(f"ghost_vp {ghost_vp} not found in ghost_aug_pos or node_pos")
                             ghost_pos = list(gmap.node_pos.values())[0]
                    _, front_vp = gmap.front_to_ghost_dist(ghost_vp)
                    front_pos = gmap.node_pos[front_vp]
                    if self.config.VIDEO_OPTION:
                        teacher_action_cpu = teacher_actions[i].cpu().item()
                        if teacher_action_cpu in [0, -100]:
                            teacher_ghost = None
                        else:
                            teacher_ghost = gmap.ghost_aug_pos[nav_inputs['gmap_vp_ids'][i][teacher_action_cpu]]
                        pred_pos = None
                        
                        vis_info = {
                            'nodes': list(gmap.node_pos.values()),
                            'ghosts': list(gmap.ghost_aug_pos.values()),
                            'predict_ghost': ghost_pos,
                            'teacher_ghost': teacher_ghost,
                            'imagine_path': None,  # Add path points predicted by imagine
                            'pred_position': pred_pos,  # Add predicted current position
                        }
                    else:
                        vis_info = None
                    # teleport to front, then forward to ghost
                    if self.config.IL.back_algo == 'control':
                        back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][front_vp]]
                        back_path = back_path[1:]
                    else:
                        back_path = None
                    env_actions.append(
                        {
                            'action': {
                                'act': 4,
                                'cur_vp': cur_vp[i],
                                'front_vp': front_vp, 'front_pos': front_pos,
                                'ghost_vp': ghost_vp, 'ghost_pos': ghost_pos,
                                'back_path': back_path,
                                'tryout': use_tryout,
                            },
                            'vis_info': vis_info,
                        }
                    )
                    prev_vp[i] = front_vp
                    # 🔥 Fix KeyError: only true ghost nodes can call delete_ghost
                    if self.config.MODEL.consume_ghost:
                        # Check if it's really a ghost node (starts with 'g')
                        if isinstance(ghost_vp, str) and ghost_vp.startswith('g'):
                            gmap.delete_ghost(ghost_vp)
                        else:
                            logger.warning(f"[KeyError Protection] Step {stepk}: Skipping deletion of non-ghost node: {ghost_vp}")

            # TTA: save current step info at the end of each step for next step usage
            # Note: if in eval mode and TTA update performed, prev_nav_inputs updated in TTA section
            # Save information only if TTA update was not performed
            
            # Save current nav_probs and gmap_vp_ids for next TTA step
            #prev_nav_probs = nav_probs.clone()
            prev_gmap_vp_ids = curr_gmap_vp_ids
            
            # NAV token: update prev_ghost_vp_ids to current ghosts
            # Must update before env.step() as number of environments may change
            if mode == 'train':
                for i in range(len(nav_inputs['gmap_vp_ids'])):
                    current_ghosts = [vp for vp in nav_inputs['gmap_vp_ids'][i] if isinstance(vp, str) and vp.startswith('g')]
                    prev_ghost_vp_ids[i] = current_ghosts

            outputs = self.envs.step(env_actions)
            num_st += 1 #new----------------------------------------------------------------
            
            observation_package, _, dones, infos = [list(x) for x in zip(*outputs)]
            
            observations, turn_state, turn_observations = [],[],[]
            for item in observation_package:
                item_1,item_2,item_3 = item
                observations.append(item_1)
                turn_state.append(item_2)
                turn_observations.append(item_3)

            # calculate metric
            if mode == 'eval':
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    gt_path = np.array(self.gt_data[str(ep_id)]['locations']).astype(np.float)
                    pred_path = np.array(info['position']['position'])
                    distances = np.array(info['position']['distance'])  
                    metric = {}
                    metric['steps_taken'] = info['steps_taken']
                    metric['distance_to_goal'] = distances[-1]
                    metric['success'] = 1. if distances[-1] <= 3. else 0.
                    metric['oracle_success'] = 1. if (distances <= 3.).any() else 0.
                    metric['path_length'] = float(np.linalg.norm(pred_path[1:] - pred_path[:-1],axis=1).sum())
                    metric['collisions'] = info['collisions']['count'] / len(pred_path)
                    gt_length = distances[0]
                    metric['spl'] = metric['success'] * gt_length / max(gt_length, metric['path_length'])
                    dtw_distance = fastdtw(pred_path, gt_path, dist=NDTW.euclidean_distance)[0]
                    metric['ndtw'] = np.exp(-dtw_distance / (len(gt_path) * 3.))
                    metric['sdtw'] = metric['ndtw'] * metric['success']
                    metric['ghost_cnt'] = self.gmaps[i].ghost_cnt
                    print(metric['oracle_success'],metric['success'],metric['spl'])
                    self.stat_eps[ep_id] = metric
                    self.pbar.update()

            # record path
            if mode == 'infer':
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    self.path_eps[ep_id] = [
                        {
                            'position': info['position_infer']['position'][0],
                            'heading': info['position_infer']['heading'][0],
                            'stop': False
                        }
                    ]
                    for p, h in zip(info['position_infer']['position'][1:], info['position_infer']['heading'][1:]):
                        if p != self.path_eps[ep_id][-1]['position']:
                            self.path_eps[ep_id].append({
                                'position': p,
                                'heading': h,
                                'stop': False
                            })
                    self.path_eps[ep_id] = self.path_eps[ep_id][:500]
                    self.path_eps[ep_id][-1]['stop'] = True
                    self.pbar.update()

            # pause env
            if sum(dones) > 0:
                for i in reversed(list(range(len(dones)))):
                    if dones[i]:
                        not_done_index.pop(i)
                        self.envs.pause_at(i)
                        observations.pop(i)
                        sg_map_global.pop(i)
                        abs_poses.pop(i)
                        positions.pop(i)
                        headings.pop(i)
                        turn_state.pop(i)
                        turn_observations.pop(i)
                        policy_net.global_fts.pop(i)
                        policy_net.global_position_x.pop(i)
                        policy_net.global_position_y.pop(i)
                        policy_net.global_position_z.pop(i)
                        policy_net.global_patch_scales.pop(i)
                        policy_net.global_patch_directions.pop(i)
                        policy_net.global_mask.pop(i)

                        # graph stop
                        self.gmaps.pop(i)
                        prev_vp.pop(i)
                        
                        # World model: clean up prev_V_pred and prev_actual_action
                        if mode == 'train':
                            prev_ghost_vp_ids.pop(i)
                            prev_V_pred.pop(i)
                            prev_actual_action.pop(i)
                        
                        # TTA: clean up TTA-related data
                        if mode == 'eval':
                            # Clean TTA history data
                            pass
                        
                        # Fix potential BUG 5: clean up memory-related data
                        if hasattr(self, 'real_obser_seq_batch') and i < len(self.real_obser_seq_batch):
                            self.real_obser_seq_batch.pop(i)
                        
                        # Fix BUG: clean up imagine history data in train mode
                        if mode == 'train':
                            pass
                        
                # Fix critical BUG 1: move outside loop to avoid repeated execution
                prev_positions_tensor = prev_positions_tensor[not_done_index]
                
                # TTA: update prev_nav_probs and prev_gmap_vp_ids, remove finished environment data
                if mode == 'eval':
                    if prev_nav_probs is not None:
                        prev_nav_probs = prev_nav_probs[not_done_index]
                    if prev_gmap_vp_ids is not None:
                        prev_gmap_vp_ids = [prev_gmap_vp_ids[i] for i in range(len(prev_gmap_vp_ids)) if i in not_done_index]
            
            if self.envs.num_envs == 0:
                break

            # obs for next step
            observations = extract_instruction_tokens(observations,self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        #exit()
        #if self.world_size > 1:
        #    torch.distributed.barrier()
        # decay = 0.2

        if mode == 'train':
            # Main navigation loss (normalized based on total_actions)
            # Numerical stability check: avoid division by zero
            if total_actions > 0:
                loss = ml_weight * loss.float() * 1.0 / total_actions
            else:
                print(f"\n[WARNING] total_actions is 0, skipping loss normalization")
                loss = ml_weight * loss.float() * 1.0
            
            self.loss += loss
            loss_cross_entropy+= loss
            # Cross entropy loss (navigation loss, already normalized and added to self.loss above)
            if loss > 0:
                print(f"loss_cross_entropy (navigation cross entropy): {loss:.4f}")
                self.logs['loss_cross_entropy'].append(loss.item())
            
            # ===== Memory Reconstruction Loss Processing =====
            memory_recon_loss_weight = 1.0
            memory_recon_loss_normalized = torch.tensor(0.0, device=self.device)
            
            if count_memory_recon > 0:
                avg_memory_recon = loss_memory_recon / count_memory_recon
                memory_recon_loss_normalized = memory_recon_loss_weight * avg_memory_recon
                print(f"loss_memory_recon (Memory reconstruction loss): average: {avg_memory_recon.item():.4f}, "
                      f"weighted: {memory_recon_loss_normalized.item():.4f}, "
                      f"accumulated: {loss_memory_recon.item():.4f}, steps: {count_memory_recon}")
                # Add to total loss
                self.loss += memory_recon_loss_normalized
                self.logs['loss_memory_recon'].append(avg_memory_recon.item())
            
            # ===== World Model Loss Processing (Added) =====
            # 1. Enhanced scoring loss
            enhanced_sap_loss_weight = 1.0  # Weight adjustable
            enhanced_sap_loss_normalized = torch.tensor(0.0, device=self.device)
            
            if count_enhanced_sap > 0:
                # Use the same normalization as the first scoring
                if total_actions > 0:
                    avg_enhanced_sap = loss_enhanced_sap / total_actions
                else:
                    avg_enhanced_sap = loss_enhanced_sap / count_enhanced_sap
                enhanced_sap_loss_normalized = enhanced_sap_loss_weight * avg_enhanced_sap * ml_weight
                print(f"loss_enhanced_sap (enhanced scoring loss): average: {avg_enhanced_sap.item():.4f}, "
                      f"weighted: {enhanced_sap_loss_normalized.item():.4f}, steps: {count_enhanced_sap}")
                self.loss += enhanced_sap_loss_normalized
                if 'loss_enhanced_sap' not in self.logs:
                    self.logs['loss_enhanced_sap'] = []
                self.logs['loss_enhanced_sap'].append(enhanced_sap_loss_normalized.item())
            
            # 2. Visual prediction loss
            vis_pred_loss_weight = 1.0  # Weight adjustable (auxiliary task, low weight)
            vis_pred_loss_normalized = torch.tensor(0.0, device=self.device)
            
            if count_vis_pred > 0:
                avg_vis_pred = loss_vis_pred / count_vis_pred
                vis_pred_loss_normalized = vis_pred_loss_weight * avg_vis_pred * ml_weight
                print(f"loss_vis_pred (visual prediction loss): average: {avg_vis_pred.item():.4f}, "
                      f"weighted: {vis_pred_loss_normalized.item():.4f}, steps: {count_vis_pred}")
                self.loss += vis_pred_loss_normalized
                if 'loss_vis_pred' not in self.logs:
                    self.logs['loss_vis_pred'] = []
                self.logs['loss_vis_pred'].append(vis_pred_loss_normalized.item())
            
            # ===== Save loss info to log file every 100 episodes =====
            if (global_iter - 2) % 100 == 0:
                import os
                log_dir = "data/logs/print"
                os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(log_dir, f"{global_iter}.txt")
                with open(log_file, 'a') as f:
                    f.write(f"\n===== Episode {global_iter} Loss Summary =====\n")
                    if loss > 0:
                        f.write(f"loss_cross_entropy (navigation cross entropy): {loss:.4f}\n")
                    if count_memory_recon > 0:
                        f.write(f"loss_memory_recon (Memory reconstruction loss): average: {(loss_memory_recon / count_memory_recon).item():.4f}, "
                                f"weighted: {memory_recon_loss_normalized.item():.4f}, steps: {count_memory_recon}\n")
                    if count_enhanced_sap > 0:
                        f.write(f"loss_enhanced_sap (enhanced scoring loss): average: {avg_enhanced_sap.item():.4f}, "
                                f"weighted: {enhanced_sap_loss_normalized.item():.4f}, steps: {count_enhanced_sap}\n")
                    if count_vis_pred > 0:
                        f.write(f"loss_vis_pred (visual prediction loss): average: {avg_vis_pred.item():.4f}, "
                                f"weighted: {vis_pred_loss_normalized.item():.4f}, steps: {count_vis_pred}\n")