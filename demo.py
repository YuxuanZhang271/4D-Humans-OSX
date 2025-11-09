import argparse
import cv2
import json
import numpy as np
import os

import torch
if torch.cuda.is_available(): 
    device = torch.device('cuda')
# elif torch.backends.mps.is_available(): 
#     device = torch.device('mps')
else: 
    device = torch.device('cpu')

from pathlib import Path

from detectron2 import model_zoo
from detectron2.config import get_cfg
from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.utils.renderer import Renderer, cam_crop_to_full
from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy

LIGHT_BLUE = (0.65098039,  0.74117647,  0.85882353)


def main(args):
    # Download and load checkpoints
    download_models(CACHE_DIR_4DHUMANS)
    model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
    # Setup HMR2.0 model
    # move model to the selected device (prefer CUDA, then MPS, then CPU)
    model = model.to(device)
    model.eval()
    # Load detector
    detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
    detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
    detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
    detector = DefaultPredictor_Lazy(detectron2_cfg)
    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.smpl.faces)

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    cap = cv2.VideoCapture(args.video_path)

    frame = 0
    transl_list = []
    global_orient_list = []
    body_pose_list = []
    betas_list = []
    focal_length_list = []
    while cap.isOpened():
        print("Current Frame: ", frame)
        ret, img_cv2 = cap.read()
        if ret:
            # Detect humans in image
            det_out = detector(img_cv2)

            det_instances = det_out['instances']
            valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
            boxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

            # Run HMR2.0 on all detected humans
            dataset = ViTDetDataset(model_cfg, img_cv2, boxes)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

            for batch in dataloader:
                batch = recursive_to(batch, device)
                with torch.no_grad():
                    out = model(batch)

                pred_cam = out['pred_cam']
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()
                scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

                batch_size = batch['img'].shape[0]
                for n in range(batch_size):
                    person_id = int(batch['personid'][n])
                    input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                    input_patch = input_patch.permute(1,2,0).numpy()

                    if isinstance(out['pred_smpl_params'], dict): 
                        person_smpl = { 
                            key: value[n].detach().cpu().numpy().tolist() 
                            for key, value in out['pred_smpl_params'].items() 
                        } 
                    else: 
                        person_smpl = out['pred_smpl_params'][n].detach().cpu().numpy().tolist()
                    
                    transl_list.append(pred_cam_t_full.tolist())
                    global_orient_list.append(person_smpl['global_orient'])
                    body_pose_list.append(person_smpl['body_pose'])
                    betas_list.append(person_smpl['betas'])
                    focal_length_list.append(float(scaled_focal_length))

        frame += 1
    
    output = {
        'transl': transl_list, 
        'global_orient': global_orient_list, 
        'body_pose': body_pose_list, 
        'betas': betas_list, 
        'focal_length': focal_length_list
    }
    json_path = os.path.join(args.out_folder, f'output.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HMR2 demo code')
    parser.add_argument('--video_path', type=str, default='example_data/images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='demo_out', help='Output folder to save rendered results')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    args = parser.parse_args()

    main(args)
