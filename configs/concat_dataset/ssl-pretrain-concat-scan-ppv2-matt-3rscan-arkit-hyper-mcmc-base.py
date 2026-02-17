_base_ = [
    "../_base_/default_runtime.py",
]

# misc custom setting
debug = 0
gpu_nums = 1 if debug else 3
batch_size = 6 * gpu_nums
batch_size_val = 6 * gpu_nums
batch_size_test = 1 * gpu_nums
num_worker = 24 * gpu_nums if not debug else 0
mix_prob = 0. # no mixup for ssl
empty_cache = False
enable_amp = True
evaluate = False
find_unused_parameters=True 

# model settings
# Trainer
train = dict(type="DefaultSSLPreTrainer")

model = dict(
    type="DefaultContrastiverSimDinoV2", 
    backbone_out_channels=512,
    local_crop_num=3, 
    do_ema=False, # use ema for training
    do_ibot=False, # if do ibot loss
    enable_mae_loss=True, # enable mae for training stability
    dino_weight=1.0,
    code_weight=1.0,
    ibot_weight=1.0,
    mask_ratio_min_max=(0.1, 0.5), # mask ratio for masked token
    mask_sample_probability=0.5, # mask probability for masked global token
    backbone=dict(
        type="PT-v3m1-simdino",
        in_channels=11,  #   all gaussian:"coord",3  "color" 3, "opacity" 1 , "quat" 4 , "scale" 3
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
        pooling_reduce="max",
    ),
)

# scheduler settings
eval_epoch = 100
epoch = eval_epoch * 8
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.001, eps=1e-4) # TODO dynamic change weight decay
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.001, 0.0001],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0001)]

# dataset settings
dataset_type = "GenericGSDataset"
data_root = (
    "/insait/qimaqi/data/scannetpp_v2_fixed_preprocessed_nochunk/"
)


hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=None),
]

# TODO too long, refine
scannetppv2_root = 'your_scannetppv2_root'  # please set your scannetppv2 root path
scannet_root = 'your_scannet_root'  # please set your scannet root path
threer_scans_root = 'your_3r_scans_root'  # please set your 3r scans root path
arkitscenes_root = 'your_arkitscenes_root'  # please set your arkitscenes root path
hypersim_root = 'your_hypersim_root'  # please set your hypersim root path
matterport_root = 'your_matterport_root'  # please set your matterport root path


data = dict(
    num_classes=100,
    ignore_index=-1,
    train=dict(
        type="ConcatDataset",
        datasets=[
            # scannetppv2 datasset
            dict(
                type='GenericGSDataset',
                split='train_grid1.0cm_chunk6x6_stride3x3',
                data_root=scannetppv2_root,
                transform=[
                    dict(type="CenterShift", apply_z=True),
                    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    dict(type="ElasticDistortion", distortion_params=[[0.9, 0.1]]), # smooth distortion
                    dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
                    dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
                    dict(type="ChromaticJitter", p=0.95, std=0.05), # no color feature augmentation here
                    dict(
                        type="GridSample",
                        grid_size=0.02,
                        hash_type="fnv",
                        mode="train",
                        keys=( "coord", "color", "scale", "quat", "opacity", ),
                        return_grid_coord=False,
                    ),
                    dict(type="SphereCrop", point_max=204800*4, mode="random"), # with this crop we will have more similarity between global and local crop
                    dict(
                        type="ContrastiveViewsGenerator_SSL",
                        local_crop_num=3, # consistent with above model config
                        view_keys=("coord", "color", "scale", "quat", "opacity", ),
                        global_base_transform=[
                            dict(type="CenterShift", apply_z=False),
                            dict(
                                type="RandomFlip",
                                p=0.5,
                            ),
                            dict(type="SphereCropRandomMaxPoints", random_scale=(0.4, 1.0), point_max=256000),
                        ],
                        local_base_transform=[
                            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                            dict(type="CenterShift", apply_z=False),
                            dict(
                                type="RandomFlip",
                                p=0.5,
                            ),
                            dict(type="SphereCropRandomMaxPoints", random_scale=(0.1, 0.4), point_max=256000), # technically, we should make a random point max here
                        ],
                        global_transform0=[
                            # for mae, in global_transform0 we use weak augmentation
                            dict(
                                type="RandomColorJitter",
                                brightness=0.4,
                                contrast=0.4,
                                saturation=0.2,
                                hue=0.1,
                                p=0.8,
                            ),
                            dict(
                                type="RandomColorGrayScale",
                                p=0.2,
                            ),
                            dict(
                                type="GridSample",
                                grid_size=0.02,
                                hash_type="fnv",
                                mode="train",
                                keys=( "coord", "color", "scale", "quat", "opacity", ),
                                return_grid_coord=True,
                            ),
                            dict(type="CenterShift", apply_z=False),
                            dict(type="NormalizeColor"),
                            dict(type="ToTensor"),
                            ],
                        global_transform1=[
                            # for mae, in global_transform1 we use stronger augmentation
                            dict(
                                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
                            ),
                            dict(
                                type="RandomColorJitter",
                                brightness=0.4,
                                contrast=0.4,
                                saturation=0.2,
                                hue=0.1,
                                p=0.8, # for mae we drcrease the probability of color jittering, to make it more stable
                            ),
                            dict(
                                type="RandomColorGrayScale",
                                p=0.2,
                            ),
                            dict(
                                type="GridSample",
                                grid_size=0.02,
                                hash_type="fnv",
                                mode="train",
                                keys=( "coord", "color", "scale", "quat", "opacity", ),
                                return_grid_coord=True,
                            ),
                            dict(type="CenterShift", apply_z=False),
                            dict(type="RandomColorSolarize",
                                p=0.2),
                            dict(type="NormalizeColor"),
                            dict(type="ToTensor"),
                            ],
                        local_transform=[
                            dict(
                                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
                            ),
                            dict(
                                type="RandomColorJitter",
                                brightness=0.4,
                                contrast=0.4,
                                saturation=0.2,
                                hue=0.1,
                                p=0.8,
                            ),
                            dict(
                                type="RandomColorGrayScale",
                                p=0.2,
                            ),
                            dict(
                                type="GridSample",
                                grid_size=0.02,
                                hash_type="fnv",
                                mode="train",
                                keys=( "coord", "color", "scale", "quat", "opacity", ),
                                return_grid_coord=True,
                            ),
                            dict(type="CenterShift", apply_z=False),
                            dict(type="NormalizeColor"),
                            dict(type="ToTensor"),
                            ],
                    ),
                    dict(
                        type="CollectContrast",
                        keys_prefix=(
                            "global_crop0",
                            "global_crop1",
                            "local_crop0",
                            "local_crop1",
                            "local_crop2",
                        ),
                        offset_keys_dict=dict(
                            global_crop0_offset="global_crop0_coord", 
                            global_crop1_offset="global_crop1_coord",
                            local_crop0_offset="local_crop0_coord", 
                            local_crop1_offset="local_crop1_coord",
                            local_crop2_offset="local_crop2_coord", 
                        ),
                        global_crop0_feat_keys=("global_crop0_color", "global_crop0_opacity", "global_crop0_quat", "global_crop0_scale", ),
                        global_crop1_feat_keys=("global_crop1_color", "global_crop1_opacity", "global_crop1_quat", "global_crop1_scale", ),
                        local_crop0_feat_keys=("local_crop0_color", "local_crop0_opacity", "local_crop0_quat", "local_crop0_scale"),
                        local_crop1_feat_keys=("local_crop1_color", "local_crop1_opacity", "local_crop1_quat", "local_crop1_scale", ),
                        local_crop2_feat_keys=("local_crop2_color", "local_crop2_opacity", "local_crop2_quat", "local_crop2_scale", ),
                        )
                    ],
                test_mode=False,
                loop=1, # focus more on benchmar dataset
            ),
            # scannet dataset:
            dict(
                    type="GenericGSDataset",
                    split="train_grid1.0cm_chunk6x6_stride3x3", # split do not matter here
                    data_root= scannet_root,
                transform=[
                    dict(type="CenterShift", apply_z=True),

                    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    dict(type="ElasticDistortion", distortion_params=[[0.9, 0.1]]), # smooth distortion
                    dict(
                        type="GridSample",
                        grid_size=0.02,
                        hash_type="fnv",
                        mode="train",
                        keys=( "coord", "color", "scale", "quat", "opacity", ),
                        return_grid_coord=False,
                    ),
                    dict(type="SphereCrop", point_max=204800*4, mode="random"), # with this crop we will have more similarity between global and local crop
                    dict(
                        type="ContrastiveViewsGenerator_SSL",
                        local_crop_num=3, # consistent with above model config
                        view_keys=("coord", "color", "scale", "quat", "opacity", ),
                        global_base_transform=[
                            dict(type="CenterShift", apply_z=False),
                            dict(
                                type="RandomFlip",
                                p=0.5,
                            ),
                            dict(type="SphereCropRandomMaxPoints", random_scale=(0.4, 1.0), point_max=256000),
                        ],
                        local_base_transform=[
                            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                            dict(type="CenterShift", apply_z=False),
                            dict(
                                type="RandomFlip",
                                p=0.5,
                            ),
                            dict(type="SphereCropRandomMaxPoints", random_scale=(0.1, 0.4), point_max=256000), # technically, we should make a random point max here
                        ],
                        global_transform0=[
                            dict(
                                type="GridSample",
                                grid_size=0.02,
                                hash_type="fnv",
                                mode="train",
                                keys=( "coord", "color", "scale", "quat", "opacity", ),
                                return_grid_coord=True,
                            ),
                            dict(type="CenterShift", apply_z=False),
                            dict(type="NormalizeColor"),
                            dict(type="ToTensor"),
                            ],
                        global_transform1=[
                            # for mae, in global_transform1 we use stronger augmentation
                            dict(
                                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
                            ),
                            dict(
                                type="RandomColorJitter",
                                brightness=0.4,
                                contrast=0.4,
                                saturation=0.2,
                                hue=0.1,
                                p=0.8, # for mae we drcrease the probability of color jittering, to make it more stable
                            ),
                            dict(
                                type="RandomColorGrayScale",
                                p=0.2,
                            ),
                            dict(
                                type="GridSample",
                                grid_size=0.02,
                                hash_type="fnv",
                                mode="train",
                                keys=( "coord", "color", "scale", "quat", "opacity", ),
                                return_grid_coord=True,
                            ),
                            dict(type="CenterShift", apply_z=False),
                            dict(type="RandomColorSolarize",
                                p=0.2),
                            dict(type="NormalizeColor"),
                            dict(type="ToTensor"),
                            ],
                        local_transform=[
                            dict(
                                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
                            ),
                            dict(
                                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
                            ),
                            dict(
                                type="RandomColorJitter",
                                brightness=0.4,
                                contrast=0.4,
                                saturation=0.2,
                                hue=0.1,
                                p=0.8,
                            ),
                            dict(
                                type="RandomColorGrayScale",
                                p=0.2,
                            ),
                            dict(
                                type="GridSample",
                                grid_size=0.02,
                                hash_type="fnv",
                                mode="train",
                                keys=( "coord", "color", "scale", "quat", "opacity", ),
                                return_grid_coord=True,
                            ),
                            dict(type="CenterShift", apply_z=False),
                            dict(type="NormalizeColor"),
                            dict(type="ToTensor"),
                            ],
                    ),
                    dict(
                        type="CollectContrast",
                        keys_prefix=(
                            "global_crop0",
                            "global_crop1",
                            "local_crop0",
                            "local_crop1",
                            "local_crop2",

                        ),
                        offset_keys_dict=dict(
                            global_crop0_offset="global_crop0_coord", 
                            global_crop1_offset="global_crop1_coord",
                            local_crop0_offset="local_crop0_coord", 
                            local_crop1_offset="local_crop1_coord",
                            local_crop2_offset="local_crop2_coord", 
                        ),
                        global_crop0_feat_keys=("global_crop0_color", "global_crop0_opacity", "global_crop0_quat", "global_crop0_scale", ),
                        global_crop1_feat_keys=("global_crop1_color", "global_crop1_opacity", "global_crop1_quat", "global_crop1_scale", ),
                        local_crop0_feat_keys=("local_crop0_color", "local_crop0_opacity", "local_crop0_quat", "local_crop0_scale"),
                        local_crop1_feat_keys=("local_crop1_color", "local_crop1_opacity", "local_crop1_quat", "local_crop1_scale", ),
                        local_crop2_feat_keys=("local_crop2_color", "local_crop2_opacity", "local_crop2_quat", "local_crop2_scale", ),
                        )
                    ],
                test_mode=False,
                loop=1, # focus more on benchmar dataset
            ),    
            #  3r scans
            dict(
                    type="GenericGSDataset",
                    split="train", # split do not matter here
                    data_root= threer_scans_root,
                transform=[
                    dict(type="CenterShift", apply_z=True),

                    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    dict(type="ElasticDistortion", distortion_params=[[0.9, 0.1]]), # smooth distortion
                    dict(
                        type="GridSample",
                        grid_size=0.02,
                        hash_type="fnv",
                        mode="train",
                        keys=( "coord", "color", "scale", "quat", "opacity", ),
                        return_grid_coord=False,
                    ),
                    dict(type="SphereCrop", point_max=204800*4, mode="random"), # with this crop we will have more similarity between global and local crop
                    dict(
                        type="ContrastiveViewsGenerator_SSL",
                        local_crop_num=3, # consistent with above model config
                        view_keys=("coord", "color", "scale", "quat", "opacity", ),
                        global_base_transform=[
                            dict(type="CenterShift", apply_z=False),
                            dict(
                                type="RandomFlip",
                                p=0.5,
                            ),
                            dict(type="SphereCropRandomMaxPoints", random_scale=(0.4, 1.0), point_max=256000),
                        ],
                        local_base_transform=[
                            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                            dict(type="CenterShift", apply_z=False),
                            dict(
                                type="RandomFlip",
                                p=0.5,
                            ),
                            dict(type="SphereCropRandomMaxPoints", random_scale=(0.1, 0.4), point_max=256000), # technically, we should make a random point max here
                        ],
                        global_transform0=[
                            dict(
                                type="GridSample",
                                grid_size=0.02,
                                hash_type="fnv",
                                mode="train",
                                keys=( "coord", "color", "scale", "quat", "opacity", ),
                                return_grid_coord=True,
                            ),
                            dict(type="CenterShift", apply_z=False),
                            dict(type="NormalizeColor"),
                            dict(type="ToTensor"),
                            ],
                        global_transform1=[
                            # for mae, in global_transform1 we use stronger augmentation
                            dict(
                                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
                            ),
                            dict(
                                type="RandomColorJitter",
                                brightness=0.4,
                                contrast=0.4,
                                saturation=0.2,
                                hue=0.1,
                                p=0.8, # for mae we drcrease the probability of color jittering, to make it more stable
                            ),
                            dict(
                                type="RandomColorGrayScale",
                                p=0.2,
                            ),
                            dict(
                                type="GridSample",
                                grid_size=0.02,
                                hash_type="fnv",
                                mode="train",
                                keys=( "coord", "color", "scale", "quat", "opacity", ),
                                return_grid_coord=True,
                            ),
                            dict(type="CenterShift", apply_z=False),
                            dict(type="RandomColorSolarize",
                                p=0.2),
                            dict(type="NormalizeColor"),
                            dict(type="ToTensor"),
                            ],
                        local_transform=[
                            dict(
                                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
                            ),
                            dict(
                                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
                            ),
                            dict(
                                type="RandomColorJitter",
                                brightness=0.4,
                                contrast=0.4,
                                saturation=0.2,
                                hue=0.1,
                                p=0.8,
                            ),
                            dict(
                                type="RandomColorGrayScale",
                                p=0.2,
                            ),
                            dict(
                                type="GridSample",
                                grid_size=0.02,
                                hash_type="fnv",
                                mode="train",
                                keys=( "coord", "color", "scale", "quat", "opacity", ),
                                return_grid_coord=True,
                            ),
                            dict(type="CenterShift", apply_z=False),
                            dict(type="NormalizeColor"),
                            dict(type="ToTensor"),
                            ],
                    ),
                    dict(
                        type="CollectContrast",
                        keys_prefix=(
                            "global_crop0",
                            "global_crop1",
                            "local_crop0",
                            "local_crop1",
                            "local_crop2",

                        ),
                        offset_keys_dict=dict(
                            global_crop0_offset="global_crop0_coord", 
                            global_crop1_offset="global_crop1_coord",
                            local_crop0_offset="local_crop0_coord", 
                            local_crop1_offset="local_crop1_coord",
                            local_crop2_offset="local_crop2_coord", 
                        ),
                        global_crop0_feat_keys=("global_crop0_color", "global_crop0_opacity", "global_crop0_quat", "global_crop0_scale", ),
                        global_crop1_feat_keys=("global_crop1_color", "global_crop1_opacity", "global_crop1_quat", "global_crop1_scale", ),
                        local_crop0_feat_keys=("local_crop0_color", "local_crop0_opacity", "local_crop0_quat", "local_crop0_scale"),
                        local_crop1_feat_keys=("local_crop1_color", "local_crop1_opacity", "local_crop1_quat", "local_crop1_scale", ),
                        local_crop2_feat_keys=("local_crop2_color", "local_crop2_opacity", "local_crop2_quat", "local_crop2_scale", ),
                        )
                    ],
                test_mode=False,
                loop=1, # focus more on benchmar dataset
            ),  
            # arkitscenes
            dict(
                    type="GenericGSDataset",
                    split="train", # split do not matter here
                    data_root= arkitscenes_root,
                transform=[
                    dict(type="CenterShift", apply_z=True),

                    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    dict(type="ElasticDistortion", distortion_params=[[0.9, 0.1]]), # smooth distortion
                    dict(
                        type="GridSample",
                        grid_size=0.02,
                        hash_type="fnv",
                        mode="train",
                        keys=( "coord", "color", "scale", "quat", "opacity", ),
                        return_grid_coord=False,
                    ),
                    dict(type="SphereCrop", point_max=204800*4, mode="random"), # with this crop we will have more similarity between global and local crop
                    dict(
                        type="ContrastiveViewsGenerator_SSL",
                        local_crop_num=3, # consistent with above model config
                        view_keys=("coord", "color", "scale", "quat", "opacity", ),
                        global_base_transform=[
                            dict(type="CenterShift", apply_z=False),
                            dict(
                                type="RandomFlip",
                                p=0.5,
                            ),
                            dict(type="SphereCropRandomMaxPoints", random_scale=(0.4, 1.0), point_max=256000),
                        ],
                        local_base_transform=[
                            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                            dict(type="CenterShift", apply_z=False),
                            dict(
                                type="RandomFlip",
                                p=0.5,
                            ),
                            dict(type="SphereCropRandomMaxPoints", random_scale=(0.1, 0.4), point_max=256000), # technically, we should make a random point max here
                        ],
                        global_transform0=[
                            dict(
                                type="GridSample",
                                grid_size=0.02,
                                hash_type="fnv",
                                mode="train",
                                keys=( "coord", "color", "scale", "quat", "opacity", ),
                                return_grid_coord=True,
                            ),
                            dict(type="CenterShift", apply_z=False),
                            dict(type="NormalizeColor"),
                            dict(type="ToTensor"),
                            ],
                        global_transform1=[
                            # for mae, in global_transform1 we use stronger augmentation
                            dict(
                                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
                            ),
                            dict(
                                type="RandomColorJitter",
                                brightness=0.4,
                                contrast=0.4,
                                saturation=0.2,
                                hue=0.1,
                                p=0.8, # for mae we drcrease the probability of color jittering, to make it more stable
                            ),
                            dict(
                                type="RandomColorGrayScale",
                                p=0.2,
                            ),
                            dict(
                                type="GridSample",
                                grid_size=0.02,
                                hash_type="fnv",
                                mode="train",
                                keys=( "coord", "color", "scale", "quat", "opacity", ),
                                return_grid_coord=True,
                            ),
                            dict(type="CenterShift", apply_z=False),
                            dict(type="RandomColorSolarize",
                                p=0.2),
                            dict(type="NormalizeColor"),
                            dict(type="ToTensor"),
                            ],
                        local_transform=[
                            dict(
                                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
                            ),
                            dict(
                                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
                            ),
                            dict(
                                type="RandomColorJitter",
                                brightness=0.4,
                                contrast=0.4,
                                saturation=0.2,
                                hue=0.1,
                                p=0.8,
                            ),
                            dict(
                                type="RandomColorGrayScale",
                                p=0.2,
                            ),
                            dict(
                                type="GridSample",
                                grid_size=0.02,
                                hash_type="fnv",
                                mode="train",
                                keys=( "coord", "color", "scale", "quat", "opacity", ),
                                return_grid_coord=True,
                            ),
                            dict(type="CenterShift", apply_z=False),
                            dict(type="NormalizeColor"),
                            dict(type="ToTensor"),
                            ],
                    ),
                    dict(
                        type="CollectContrast",
                        keys_prefix=(
                            "global_crop0",
                            "global_crop1",
                            "local_crop0",
                            "local_crop1",
                            "local_crop2",

                        ),
                        offset_keys_dict=dict(
                            global_crop0_offset="global_crop0_coord", 
                            global_crop1_offset="global_crop1_coord",
                            local_crop0_offset="local_crop0_coord", 
                            local_crop1_offset="local_crop1_coord",
                            local_crop2_offset="local_crop2_coord", 
                        ),
                        global_crop0_feat_keys=("global_crop0_color", "global_crop0_opacity", "global_crop0_quat", "global_crop0_scale", ),
                        global_crop1_feat_keys=("global_crop1_color", "global_crop1_opacity", "global_crop1_quat", "global_crop1_scale", ),
                        local_crop0_feat_keys=("local_crop0_color", "local_crop0_opacity", "local_crop0_quat", "local_crop0_scale"),
                        local_crop1_feat_keys=("local_crop1_color", "local_crop1_opacity", "local_crop1_quat", "local_crop1_scale", ),
                        local_crop2_feat_keys=("local_crop2_color", "local_crop2_opacity", "local_crop2_quat", "local_crop2_scale", ),
                        )
                    ],
                test_mode=False,
                loop=1, # focus more on benchmar dataset
            ),  
            # hypersim
            dict(
                    type="GenericGSDataset",
                    split="train", # split do not matter here
                    data_root= hypersim_root,
                transform=[
                    dict(type="CenterShift", apply_z=True),

                    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    dict(type="ElasticDistortion", distortion_params=[[0.9, 0.1]]), # smooth distortion
                    dict(
                        type="GridSample",
                        grid_size=0.02,
                        hash_type="fnv",
                        mode="train",
                        keys=( "coord", "color", "scale", "quat", "opacity", ),
                        return_grid_coord=False,
                    ),
                    dict(type="SphereCrop", point_max=204800*4, mode="random"), # with this crop we will have more similarity between global and local crop
                    dict(
                        type="ContrastiveViewsGenerator_SSL",
                        local_crop_num=3, # consistent with above model config
                        view_keys=("coord", "color", "scale", "quat", "opacity", ),
                        global_base_transform=[
                            dict(type="CenterShift", apply_z=False),
                            dict(
                                type="RandomFlip",
                                p=0.5,
                            ),
                            dict(type="SphereCropRandomMaxPoints", random_scale=(0.4, 1.0), point_max=256000),
                        ],
                        local_base_transform=[
                            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                            dict(type="CenterShift", apply_z=False),
                            dict(
                                type="RandomFlip",
                                p=0.5,
                            ),
                            dict(type="SphereCropRandomMaxPoints", random_scale=(0.1, 0.4), point_max=256000), # technically, we should make a random point max here
                        ],
                        global_transform0=[
                            dict(
                                type="GridSample",
                                grid_size=0.02,
                                hash_type="fnv",
                                mode="train",
                                keys=( "coord", "color", "scale", "quat", "opacity", ),
                                return_grid_coord=True,
                            ),
                            dict(type="CenterShift", apply_z=False),
                            dict(type="NormalizeColor"),
                            dict(type="ToTensor"),
                            ],
                        global_transform1=[
                            # for mae, in global_transform1 we use stronger augmentation
                            dict(
                                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
                            ),
                            dict(
                                type="RandomColorJitter",
                                brightness=0.4,
                                contrast=0.4,
                                saturation=0.2,
                                hue=0.1,
                                p=0.8, # for mae we drcrease the probability of color jittering, to make it more stable
                            ),
                            dict(
                                type="RandomColorGrayScale",
                                p=0.2,
                            ),
                            dict(
                                type="GridSample",
                                grid_size=0.02,
                                hash_type="fnv",
                                mode="train",
                                keys=( "coord", "color", "scale", "quat", "opacity", ),
                                return_grid_coord=True,
                            ),
                            dict(type="CenterShift", apply_z=False),
                            dict(type="RandomColorSolarize",
                                p=0.2),
                            dict(type="NormalizeColor"),
                            dict(type="ToTensor"),
                            ],
                        local_transform=[
                            dict(
                                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
                            ),
                            dict(
                                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
                            ),
                            dict(
                                type="RandomColorJitter",
                                brightness=0.4,
                                contrast=0.4,
                                saturation=0.2,
                                hue=0.1,
                                p=0.8,
                            ),
                            dict(
                                type="RandomColorGrayScale",
                                p=0.2,
                            ),
                            dict(
                                type="GridSample",
                                grid_size=0.02,
                                hash_type="fnv",
                                mode="train",
                                keys=( "coord", "color", "scale", "quat", "opacity", ),
                                return_grid_coord=True,
                            ),
                            dict(type="CenterShift", apply_z=False),
                            dict(type="NormalizeColor"),
                            dict(type="ToTensor"),
                            ],
                    ),
                    dict(
                        type="CollectContrast",
                        keys_prefix=(
                            "global_crop0",
                            "global_crop1",
                            "local_crop0",
                            "local_crop1",
                            "local_crop2",

                        ),
                        offset_keys_dict=dict(
                            global_crop0_offset="global_crop0_coord", 
                            global_crop1_offset="global_crop1_coord",
                            local_crop0_offset="local_crop0_coord", 
                            local_crop1_offset="local_crop1_coord",
                            local_crop2_offset="local_crop2_coord", 
                        ),
                        global_crop0_feat_keys=("global_crop0_color", "global_crop0_opacity", "global_crop0_quat", "global_crop0_scale", ),
                        global_crop1_feat_keys=("global_crop1_color", "global_crop1_opacity", "global_crop1_quat", "global_crop1_scale", ),
                        local_crop0_feat_keys=("local_crop0_color", "local_crop0_opacity", "local_crop0_quat", "local_crop0_scale"),
                        local_crop1_feat_keys=("local_crop1_color", "local_crop1_opacity", "local_crop1_quat", "local_crop1_scale", ),
                        local_crop2_feat_keys=("local_crop2_color", "local_crop2_opacity", "local_crop2_quat", "local_crop2_scale", ),
                        )
                    ],
                test_mode=False,
                loop=1, # focus more on benchmar dataset
            ),  
            # matterport
            dict(
                    type="GenericGSDataset",
                    split="train", # split do not matter here
                    data_root= matterport_root,
                transform=[
                    dict(type="CenterShift", apply_z=True),

                    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    dict(type="ElasticDistortion", distortion_params=[[0.9, 0.1]]), # smooth distortion
                    dict(
                        type="GridSample",
                        grid_size=0.02,
                        hash_type="fnv",
                        mode="train",
                        keys=( "coord", "color", "scale", "quat", "opacity", ),
                        return_grid_coord=False,
                    ),
                    dict(type="SphereCrop", point_max=204800*4, mode="random"), # with this crop we will have more similarity between global and local crop
                    dict(
                        type="ContrastiveViewsGenerator_SSL",
                        local_crop_num=3, # consistent with above model config
                        view_keys=("coord", "color", "scale", "quat", "opacity", ),
                        global_base_transform=[
                            dict(type="CenterShift", apply_z=False),
                            dict(
                                type="RandomFlip",
                                p=0.5,
                            ),
                            dict(type="SphereCropRandomMaxPoints", random_scale=(0.4, 1.0), point_max=256000),
                        ],
                        local_base_transform=[
                            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                            dict(type="CenterShift", apply_z=False),
                            dict(
                                type="RandomFlip",
                                p=0.5,
                            ),
                            dict(type="SphereCropRandomMaxPoints", random_scale=(0.1, 0.4), point_max=256000), # technically, we should make a random point max here
                        ],
                        global_transform0=[
                            dict(
                                type="GridSample",
                                grid_size=0.02,
                                hash_type="fnv",
                                mode="train",
                                keys=( "coord", "color", "scale", "quat", "opacity", ),
                                return_grid_coord=True,
                            ),
                            dict(type="CenterShift", apply_z=False),
                            dict(type="NormalizeColor"),
                            dict(type="ToTensor"),
                            ],
                        global_transform1=[
                            # for mae, in global_transform1 we use stronger augmentation
                            dict(
                                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
                            ),
                            dict(
                                type="RandomColorJitter",
                                brightness=0.4,
                                contrast=0.4,
                                saturation=0.2,
                                hue=0.1,
                                p=0.8, # for mae we drcrease the probability of color jittering, to make it more stable
                            ),
                            dict(
                                type="RandomColorGrayScale",
                                p=0.2,
                            ),
                            dict(
                                type="GridSample",
                                grid_size=0.02,
                                hash_type="fnv",
                                mode="train",
                                keys=( "coord", "color", "scale", "quat", "opacity", ),
                                return_grid_coord=True,
                            ),
                            dict(type="CenterShift", apply_z=False),
                            dict(type="RandomColorSolarize",
                                p=0.2),
                            dict(type="NormalizeColor"),
                            dict(type="ToTensor"),
                            ],
                        local_transform=[
                            dict(
                                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
                            ),
                            dict(
                                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
                            ),
                            dict(
                                type="RandomColorJitter",
                                brightness=0.4,
                                contrast=0.4,
                                saturation=0.2,
                                hue=0.1,
                                p=0.8,
                            ),
                            dict(
                                type="RandomColorGrayScale",
                                p=0.2,
                            ),
                            dict(
                                type="GridSample",
                                grid_size=0.02,
                                hash_type="fnv",
                                mode="train",
                                keys=( "coord", "color", "scale", "quat", "opacity", ),
                                return_grid_coord=True,
                            ),
                            dict(type="CenterShift", apply_z=False),
                            dict(type="NormalizeColor"),
                            dict(type="ToTensor"),
                            ],
                    ),
                    dict(
                        type="CollectContrast",
                        keys_prefix=(
                            "global_crop0",
                            "global_crop1",
                            "local_crop0",
                            "local_crop1",
                            "local_crop2",

                        ),
                        offset_keys_dict=dict(
                            global_crop0_offset="global_crop0_coord", 
                            global_crop1_offset="global_crop1_coord",
                            local_crop0_offset="local_crop0_coord", 
                            local_crop1_offset="local_crop1_coord",
                            local_crop2_offset="local_crop2_coord", 
                        ),
                        global_crop0_feat_keys=("global_crop0_color", "global_crop0_opacity", "global_crop0_quat", "global_crop0_scale", ),
                        global_crop1_feat_keys=("global_crop1_color", "global_crop1_opacity", "global_crop1_quat", "global_crop1_scale", ),
                        local_crop0_feat_keys=("local_crop0_color", "local_crop0_opacity", "local_crop0_quat", "local_crop0_scale"),
                        local_crop1_feat_keys=("local_crop1_color", "local_crop1_opacity", "local_crop1_quat", "local_crop1_scale", ),
                        local_crop2_feat_keys=("local_crop2_color", "local_crop2_opacity", "local_crop2_quat", "local_crop2_scale", ),
                        )
                    ],
                test_mode=False,
                loop=1, # focus more on benchmar dataset
            ),  
        ]
    ),
)
