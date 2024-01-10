default_scope = 'mmaction'
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20, ignore_last=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
log_level = 'INFO'
load_from = None
resume = False
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',
        in_channels=17,
        base_channels=32,
        num_stages=3,
        out_indices=(2, ),
        stage_blocks=(
            4,
            6,
            3,
        ),
        conv1_stride_s=1,
        pool1_stride_s=1,
        inflate=(
            0,
            1,
            1,
        ),
        spatial_strides=(
            2,
            2,
            2,
        ),
        temporal_strides=(
            1,
            1,
            1,
        ),
        dilations=(
            1,
            1,
            1,
        )),
    cls_head=dict(
        type='I3DHead',
        in_channels=512,
        num_classes=2,
        dropout_ratio=0.5,
        average_clips='prob'))
dataset_type = 'PoseDataset'
ann_file = '/workspace/police_lab/mmaction2_mhncity/gen_dataset/annotations/custom_dataset/total_add.pkl'
left_kp = [
    1,
    3,
    5,
    7,
    9,
    11,
    13,
    15,
]
right_kp = [
    2,
    4,
    6,
    8,
    10,
    12,
    14,
    16,
]
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=32),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
    dict(type='Resize', scale=(
        64,
        64,
    ), keep_ratio=False),
    dict(type='RandomResizedCrop', area_range=(
        0.56,
        1.0,
    )),
    dict(type='Resize', scale=(
        56,
        56,
    ), keep_ratio=False),
    dict(
        type='Flip',
        flip_ratio=0.5,
        left_kp=[
            1,
            3,
            5,
            7,
            9,
            11,
            13,
            15,
        ],
        right_kp=[
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
        ]),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='PackActionInputs'),
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=32, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
    dict(type='Resize', scale=(
        64,
        64,
    ), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='PackActionInputs'),
]
test_pipeline = [
    dict(
        type='UniformSampleFrames', clip_len=32, num_clips=10, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
    dict(type='Resize', scale=(
        64,
        64,
    ), keep_ratio=False),
    dict(
        type='GeneratePoseTarget',
        with_kp=True,
        with_limb=False,
        left_kp=[
            1,
            3,
            5,
            7,
            9,
            11,
            13,
            15,
        ],
        right_kp=[
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
        ]),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='PackActionInputs'),
]
train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(
            type='PoseDataset',
            ann_file=
            '/workspace/police_lab/mmaction2_mhncity/gen_dataset/annotations/custom_dataset/total_add.pkl',
            split='xsub_train',
            pipeline=[
                dict(type='UniformSampleFrames', clip_len=32),
                dict(type='PoseDecode'),
                dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
                dict(type='Resize', scale=(
                    64,
                    64,
                ), keep_ratio=False),
                dict(type='RandomResizedCrop', area_range=(
                    0.56,
                    1.0,
                )),
                dict(type='Resize', scale=(
                    56,
                    56,
                ), keep_ratio=False),
                dict(
                    type='Flip',
                    flip_ratio=0.5,
                    left_kp=[
                        1,
                        3,
                        5,
                        7,
                        9,
                        11,
                        13,
                        15,
                    ],
                    right_kp=[
                        2,
                        4,
                        6,
                        8,
                        10,
                        12,
                        14,
                        16,
                    ]),
                dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
                dict(type='FormatShape', input_format='NCTHW_Heatmap'),
                dict(type='PackActionInputs'),
            ])))
val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='PoseDataset',
        ann_file=
        '/workspace/police_lab/mmaction2_mhncity/gen_dataset/annotations/custom_dataset/total_add.pkl',
        split='xsub_val',
        pipeline=[
            dict(
                type='UniformSampleFrames',
                clip_len=32,
                num_clips=1,
                test_mode=True),
            dict(type='PoseDecode'),
            dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
            dict(type='Resize', scale=(
                64,
                64,
            ), keep_ratio=False),
            dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
            dict(type='FormatShape', input_format='NCTHW_Heatmap'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='PoseDataset',
        ann_file=
        '/workspace/police_lab/mmaction2_mhncity/gen_dataset/annotations/custom_dataset/total_add.pkl',
        split='xsub_val',
        pipeline=[
            dict(
                type='UniformSampleFrames',
                clip_len=32,
                num_clips=10,
                test_mode=True),
            dict(type='PoseDecode'),
            dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
            dict(type='Resize', scale=(
                64,
                64,
            ), keep_ratio=False),
            dict(
                type='GeneratePoseTarget',
                with_kp=True,
                with_limb=False,
                left_kp=[
                    1,
                    3,
                    5,
                    7,
                    9,
                    11,
                    13,
                    15,
                ],
                right_kp=[
                    2,
                    4,
                    6,
                    8,
                    10,
                    12,
                    14,
                    16,
                ]),
            dict(type='FormatShape', input_format='NCTHW_Heatmap'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True))
val_evaluator = [
    dict(type='AccMetric'),
]
test_evaluator = [
    dict(type='AccMetric'),
]
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=18, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=0,
        T_max=18,
        by_epoch=True,
        convert_to_iter_based=True),
]
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.2, momentum=0.9, weight_decay=0.0003),
    clip_grad=dict(max_norm=40, norm_type=2))
auto_scale_lr = dict(enable=False, base_batch_size=128)
launcher = 'pytorch'
work_dir = './work_dirs/pose_only_custom'
randomness = dict(seed=None, diff_rank_seed=False, deterministic=False)
