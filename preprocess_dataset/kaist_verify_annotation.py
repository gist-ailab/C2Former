from mmcv import Config


def test_dataset_loading(cfg):
    from mmrotate.datasets import build_dataset

    # 데이터셋 빌드
    dataset = build_dataset(cfg.data.train)

    # 데이터셋 길이 출력
    print(f"Dataset length: {len(dataset)}")
    
    # 데이터셋이 비어있는지 확인
    if len(dataset) == 0:
        print("Error: Dataset is empty.")
    else:
        print(f"First sample in dataset: {dataset[0]}")

if __name__ == '__main__':
    cfg = Config.fromfile('/SSDe/heeseon/src/C2Former/configs/s2anet/s2anet_c2former_fpn_1x_kaist_le135.py')
    test_dataset_loading(cfg)
