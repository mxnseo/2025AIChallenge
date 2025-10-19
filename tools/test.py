import _init_path
import argparse
import glob
from pathlib import Path
import numpy as np
import torch
import os
from tqdm import tqdm

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
import pickle
from thop import profile
import time

os.environ["NCCL_P2P_DISABLE"] = "1" 


def get_filename_without_extension(file_path):
    """파일 전체 경로에서 확장자를 제외한 파일명을 추출합니다."""
    filename_with_extension = os.path.basename(file_path)
    filename_without_extension = os.path.splitext(filename_with_extension)[0]
    return filename_without_extension

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.npy'):
        """
        Args:
            dataset_cfg (dict): 데이터셋 설정.
            class_names (list): 클래스 이름들.
            training (bool): 학습 여부.
            root_path (Path or str): 데이터셋 루트 경로.
            logger (Logger): 로거.
            ext (str): 파일 확장자.
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext

        test_frame_ids_filename = os.path.join(root_path, "ImageSets/test.txt")
        with open(test_frame_ids_filename, 'r') as f:
            test_frame_ids = [line.strip() for line in f.readlines()]

        self.sample_file_list = sorted(
            [os.path.join(root_path, f"points/{frame_id}{ext}") for frame_id in test_frame_ids]
        )

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError(f"File extension {self.ext} is not supported")

        frame_id = get_filename_without_extension(self.sample_file_list[index])
        input_dict = {'points': points, 'frame_id': frame_id}

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def parse_config():
    parser = argparse.ArgumentParser(description='Argument parser for demo')
    parser.add_argument('--cfg_file', type=str, required=False, help='Specify the config for demo')
    parser.add_argument('--data_path', type=str, required=False, help='Specify the custom_av directory')
    parser.add_argument('--ckpt', type=str, required=False, help='Specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.npy', help='Specify the extension of your point cloud data file')

    args = parser.parse_args()
    # args.cfg_file = "cfgs/custom_av/centerpoint_pillar_1x_long_epoch.yaml"
    # args.ckpt = "../output/custom_av/centerpoint_pillar_1x_long_epoch/default/ckpt/checkpoint_epoch_80.pth"
    args.data_path = "/mnt/d/data/custom_av"

    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()

    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    dummy_data = demo_dataset.collate_batch([demo_dataset[0]])
    load_data_to_gpu(dummy_data)
    # ---- FLOPs 측정 ----
    try:
        macs, params = profile(model, inputs=(dummy_data,), verbose=False)
        flops = macs * 2 / 1e9  # GFlops로 환산
        with open("flops.txt", "w") as f:
            f.write(f"{flops:.3f}\n")
        logger.info(f"FLOPs: {flops:.3f} GFLOPs (saved to flops.txt)")
    except Exception as e:
        logger.error(f"FLOPs 측정 실패: {e}")

    # ---- GPU Warm-up ----
    for _ in range(20):  # warm-up 20회
        _ = model(dummy_data)


    save_filename = "result.pkl"

    det_annos = list() 
    with torch.no_grad():
        total_time = 0.0
        num_samples = 0

        for idx, data_dict in enumerate(tqdm(demo_dataset, desc="Processing dataset")):
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            torch.cuda.synchronize()
            start = time.time()
            pred_dicts, _ = model.forward(data_dict)
            torch.cuda.synchronize()
            end = time.time()
            elapsed = (end - start) * 1000  # ms 단위
            total_time += elapsed
            num_samples += 1
            pred_dict = pred_dicts[0]
            frame_id = data_dict['frame_id'][0]
            pred_dict = {k: v.cpu().numpy() for k, v in pred_dict.items()}

            num_obj = len(pred_dict['pred_labels'])

            if num_obj == 0:
                print("At least one object should be detected.")
                assert num_obj != 0, "No objects detected. The program requires at least one object to be detected."

            class_names = list()
            for obj_idx in range(num_obj):
                class_id = pred_dict['pred_labels'][obj_idx] - 1 
                class_name = cfg.CLASS_NAMES[class_id]
                class_names.append(class_name)
        
            det_anno = {
                'name' : np.array(class_names, dtype='<U10'),
                'score' : pred_dict['pred_scores'],
                'boxes_lidar': pred_dict['pred_boxes'],
                'pred_labels': pred_dict['pred_labels'],
                'frame_id': frame_id,
            }

            det_annos.append(det_anno)

        if num_samples > 0:
            avg_time = total_time / num_samples
            logger.info(f"Average inference time per sample: {avg_time:.3f} ms")
    
    with open(save_filename, 'wb') as f:
        pickle.dump(det_annos, f)

if __name__ == '__main__':
    main()
