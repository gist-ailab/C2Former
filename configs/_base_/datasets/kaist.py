# dataset settings
dataset_type = 'KaistDataset'  # load paired image
data_root = '/ailab_mat/dataset/KAIST_PED/kaist-cvpr15/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadPairedImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(512, 640)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='PairedImageDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'img_tir', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadPairedImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 640),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='PairedImageDefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'img_tir'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file='/SSDe/heeseon/src/C2Former/trainval.json',
        img_prefix='',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file='/SSDe/heeseon/src/C2Former/test.json',
        img_prefix='',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='/SSDe/heeseon/src/C2Former/test.json',
        img_prefix='',
        pipeline=test_pipeline))