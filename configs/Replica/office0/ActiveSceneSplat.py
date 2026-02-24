import numpy as np
import os

_base_ = "../../default.py"

##################################################
### General
##################################################
general = dict(
    dataset = "Replica",
    scene = "office0",
    num_iter = 2000,
    device = 'cuda'
)

##################################################
### Directories
##################################################
dirs = dict(
    data_dir = "data/",
    result_dir = "results/",
    cfg_dir = os.path.join("configs", general['dataset'], general['scene'])
)


##################################################
### Simulator
##################################################
sim = dict(
    method = "habitat_v2"
)

if sim["method"] == "habitat_v2":
    sim.update(
        habitat_cfg = os.path.join(dirs['cfg_dir'], "habitat.py")
    )

##################################################
### SLAM
##################################################
slam = dict(
    method="scenesplat"
)

if slam["method"] == "scenesplat":
    slam.update(
        room_cfg        = f"{dirs['cfg_dir']}/../replica_splatam_s.py",
        enable_active_planning = True,
        dataset_eval_basedir = "data/replica_sim_nvs",

        ### bounding box ###
        bbox_bound = [[-2.1,2.5],[-3.2,2],[-1.3,2.0]],
        bbox_voxel_size = 0.05,

        surface_dist_thre=0.5,
        find_free_indices_bs=1000,

        ### Refinement step ###
        refine_map_iter = 60,
        use_global_keyframe = True,
        global_keyframe = dict(
            completeness_thre = 0.1,
            color_thre = 34,
            depth_thre = 0.01,
            quality_method = "relative",
            quality_freq = 100,
            quality_perc_thre = 30,
        ),

        ##### SceneSplat Network #######
        num_topk_logits = 16,
        num_semantic_classes = 16,
        scenesplat_grid_size = 0.02,
        pointcept_path = '/mnt/Data2/liyan/SceneSplat',
        scenesplat_checkpoint = '/mnt/Data4/scene_splat_7k/checkpoints/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.pth',
        autoencoder_checkpoint = '/mnt/Data2/liyan/SceneSplat/autoencoder/ckpt/matterport/best_ckpt.pth',

        ### override ###
        override = dict(
            map_every = 5,
            report_global_progress_every = 5,
            tracking = dict(
                use_gt_poses=True,
            )
        )
    )

##################################################
### Planner
##################################################
planner = dict(
    method= "active_gsv2",

    ### active_gs params ###
    max_exploration_steps = 1500,
    post_refine_steps = 200,
    max_refinement_steps = 200,
    num_exploration_stage = 2,
    gs_z_levels = [
        [35],
        [20, 50],
    ],
    num_dir_samples = [
        5,
        15,
    ],

    xy_sampling_step = [
        1.0,
        0.5,
    ],

    trans_step_size = 0.1,
    rot_step_size = 10,

    surface_dist_thre = slam['surface_dist_thre'],
    topk_cls_confidence = [16,
                           5],

    ### Stop Criteria ###
    explore_thre = 0.005,
    recognize_thre = 0.3,
    color_ig_thre = 34,
    depth_ig_thre = 0.01,
    post_refinement_eval_freq = 100,

    up_dir = np.array([0, 0, 1]),
    use_traj_pose = True,
    SLAMData_dir = os.path.join(
        dirs["data_dir"],
        "Replica", general['scene']
        ),

    ### RRT ###
    local_planner_method = "RRTNaruto",
)

if planner["local_planner_method"] == "RRTNaruto":
    planner.update(
        rrt_step_size = planner['trans_step_size'] / slam['bbox_voxel_size'],
        rrt_step_amplifier = 10,
        rrt_maxz = 100,
        rrt_max_iter = None,
        rrt_z_levels = None,
        enable_eval = False,
        enable_direct_line = True,
    )

##################################################
### Visualization
##################################################
visualizer = dict(
    method = "active_lang",
    vis_rgbd        = True,
    vis_rgbd_max_depth = 10
)
