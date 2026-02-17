_base_ = [
    "../_base_/default_runtime.py",
]

# misc custom setting
debug = 0
gpu_nums = 1 if debug else 3
batch_size = 8 * gpu_nums
batch_size_val = 8 * gpu_nums
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
    do_ema=True, # use ema for training
    do_ibot=True, # if do ibot loss
    enable_mae_loss=True, # enable mae for training stability
    dino_weight=1.0,
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
    "/insait/qimaqi/data/scannet_fixed_preprocessed_nochunk/"
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=None),
]


data = dict(
    num_classes=200,
    ignore_index=-1,
    train=dict(
        type=dataset_type,
        split="train", # split do not matter here
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
            ),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                keys=( "coord", "color", "scale", "quat", "opacity", ),
                return_grid_coord=False,
            ),
            dict(type="SphereCrop", point_max=204800*4, mode="random"), 
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
                    dict(type="SphereCropRandomMaxPoints", random_scale=(0.4, 1.0), point_max=102400*3),
                ],
                local_base_transform=[
                    dict(type="CenterShift", apply_z=False),
                    dict(
                        type="RandomFlip",
                        p=0.5,
                    ),
                    dict(type="SphereCropRandomMaxPoints", random_scale=(0.1, 0.4), point_max=102400*3), # technically, we should make a random point max here
                ],
                global_transform0=[
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
                    dict(type="GSGaussianBlurVoxelOpc",
                         p=1.0,
                         extra_keys=("scale", "quat", "opacity", ), # for mae training we do not need such strong blur
                    ),
                    dict(type="NormalizeColor"),
                    dict(type="ToTensor"),
                    ],
                global_transform1=[
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
                    # dict(type="NormalizeGSScale"),
                    dict(type="GSGaussianBlurVoxelOpc",
                         p=0.1,
                         extra_keys=("scale", "quat", "opacity", ), # for mae training we do not need such strong blur
                    ), # in mae, we have one withour any blur
                    dict(type="RandomColorSolarize",
                         p=0.2),
                    dict(type="NormalizeColor"),
                    dict(type="ToTensor"),
                    ],
                local_transform=[
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
                    # dict(type="NormalizeGSScale"),
                    dict(type="GSGaussianBlurVoxelOpc",
                         p=0.5,
                        extra_keys=("scale", "quat", "opacity", ),
                    ),
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
    ),
)