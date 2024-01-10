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
        depth=50,
        conv1_kernel=(
            1,
            7,
            7,
        ),
        inflate=(
            0,
            0,
            1,
            1,
        )),
    cls_head=dict(
        type='I3DHead',
        in_channels=2048,
        num_classes=2,
        dropout_ratio=0.5,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        format_shape='NCTHW'))
dataset_type = 'PoseDataset'
data_root = '/workspace/police_lab/dataset/yolo_selfharm_c2_aug_vid_960/total/'
ann_file = '/workspace/police_lab/mmaction2_mhncity/gen_dataset/annotations/custom_dataset/total_add.pkl'
train_pipeline = [
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=8), num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1.0, allow_imgpad=True),
    dict(type='Resize', scale=(
        256,
        256,
    ), keep_ratio=False),
    dict(type='RandomResizedCrop', area_range=(
        0.56,
        1.0,
    )),
    dict(type='Resize', scale=(
        224,
        224,
    ), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs'),
]
file_client_args = dict(io_backend='disk')
val_pipeline = [
    dict(
        type='SampleAVAFrames', clip_len=32, frame_interval=2, test_mode=True),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs')
]
# val_pipeline = [
#     dict(
#         type='MMUniformSampleFrames',
#         clip_len=dict(RGB=8),
#         num_clips=1,
#         test_mode=True),
#     dict(type='MMDecode'),
#     dict(type='MMCompact', hw_ratio=1.0, allow_imgpad=True),
#     dict(type='Resize', scale=(
#         224,
#         224,
#     ), keep_ratio=False),
#     dict(type='FormatShape', input_format='NCTHW'),
#     dict(type='PackActionInputs'),
# ]
test_pipeline = [
    dict(
        type='MMUniformSampleFrames',
        clip_len=dict(RGB=8),
        num_clips=10,
        test_mode=True),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1.0, allow_imgpad=True),
    dict(type='Resize', scale=(
        224,
        224,
    ), keep_ratio=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs'),
]
train_dataloader = dict(
    batch_size=12,
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
            data_prefix=dict(
                video=
                '/workspace/police_lab/dataset/yolo_selfharm_c2_aug_vid_960/total/'
            ),
            split='xsub_train',
            pipeline=[
                dict(
                    type='MMUniformSampleFrames',
                    clip_len=dict(RGB=8),
                    num_clips=1),
                dict(type='MMDecode'),
                dict(type='MMCompact', hw_ratio=1.0, allow_imgpad=True),
                dict(type='Resize', scale=(
                    256,
                    256,
                ), keep_ratio=False),
                dict(type='RandomResizedCrop', area_range=(
                    0.56,
                    1.0,
                )),
                dict(type='Resize', scale=(
                    224,
                    224,
                ), keep_ratio=False),
                dict(type='Flip', flip_ratio=0.5),
                dict(type='FormatShape', input_format='NCTHW'),
                dict(type='PackActionInputs'),
            ])))
val_dataloader = dict(
    batch_size=12,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='PoseDataset',
        ann_file=
        '/workspace/police_lab/mmaction2_mhncity/gen_dataset/annotations/custom_dataset/total_add.pkl',
        data_prefix=dict(
            video=
            '/workspace/police_lab/dataset/yolo_selfharm_c2_aug_vid_960/total/'
        ),
        split='xsub_val',
        pipeline=[
            dict(
                type='MMUniformSampleFrames',
                clip_len=dict(RGB=8),
                num_clips=1,
                test_mode=True),
            dict(type='MMDecode'),
            dict(type='MMCompact', hw_ratio=1.0, allow_imgpad=True),
            dict(type='Resize', scale=(
                224,
                224,
            ), keep_ratio=False),
            dict(type='FormatShape', input_format='NCTHW'),
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
        data_prefix=dict(
            video=
            '/workspace/police_lab/dataset/yolo_selfharm_c2_aug_vid_960/total/'
        ),
        split='xsub_val',
        pipeline=[
            dict(
                type='MMUniformSampleFrames',
                clip_len=dict(RGB=8),
                num_clips=10,
                test_mode=True),
            dict(type='MMDecode'),
            dict(type='MMCompact', hw_ratio=1.0, allow_imgpad=True),
            dict(type='Resize', scale=(
                224,
                224,
            ), keep_ratio=False),
            dict(type='FormatShape', input_format='NCTHW'),
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
    optimizer=dict(type='SGD', lr=0.15, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))
auto_scale_lr = dict(enable=False, base_batch_size=96)
launcher = 'pytorch'
work_dir = './work_dirs/rgb_only_custom'
randomness = dict(seed=None, diff_rank_seed=False, deterministic=False)
