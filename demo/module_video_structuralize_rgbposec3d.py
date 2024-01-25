# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy as cp
import tempfile
import warnings

import cv2
import mmcv
import mmengine
import numpy as np
import torch
from mmengine import DictAction
from mmengine.structures import InstanceData

from mmaction.apis import (detection_inference, inference_recognizer,
                           inference_skeleton, init_recognizer, pose_inference)
from mmaction.registry import VISUALIZERS
from mmaction.structures import ActionDataSample
from mmaction.utils import frame_extract

try:
    from mmdet.apis import init_detector
except (ImportError, ModuleNotFoundError):
    warnings.warn('Failed to import `init_detector` form `mmdet.apis`. '
                  'These apis are required in skeleton-based applications! ')

try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

#추가
from pathlib import Path
from PIL import Image
# from tracking_skeleton import detect_m

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.5
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1


def hex2color(h):
    """Convert the 6-digit hex string to tuple of 3 int value (RGB)"""
    return (int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))


PLATEBLUE = '03045e-023e8a-0077b6-0096c7-00b4d8-48cae4'
PLATEBLUE = PLATEBLUE.split('-')
PLATEBLUE = [hex2color(h) for h in PLATEBLUE]
PLATEGREEN = '004b23-006400-007200-008000-38b000-70e000'
PLATEGREEN = PLATEGREEN.split('-')
PLATEGREEN = [hex2color(h) for h in PLATEGREEN]


def visualize(args,
              frames,
              annotations,
              pose_data_samples,
              action_result,
              plate=PLATEBLUE,
              max_num=5):
    """Visualize frames with predicted annotations.

    Args:
        frames (list[np.ndarray]): Frames for visualization, note that
            len(frames) % len(annotations) should be 0.
        annotations (list[list[tuple]]): The predicted spatio-temporal
            detection results.
        pose_data_samples (list[list[PoseDataSample]): The pose results.
        action_result (str): The predicted action recognition results.
        pose_model (nn.Module): The constructed pose model.
        plate (str): The plate used for visualization. Default: PLATEBLUE.
        max_num (int): Max number of labels to visualize for a person box.
            Default: 5.

    Returns:
        list[np.ndarray]: Visualized frames.
    """

    assert max_num + 1 <= len(plate)
    frames_ = cp.deepcopy(frames)
    frames_ = [mmcv.imconvert(f, 'bgr', 'rgb') for f in frames_]
    nf, na = len(frames), len(annotations)
    assert nf % na == 0
    nfpa = len(frames) // len(annotations)
    anno = None
    h, w, _ = frames[0].shape
    scale_ratio = np.array([w, h, w, h])

    # add pose results
    if pose_data_samples:
        pose_config = mmengine.Config.fromfile(args.pose_config)
        visualizer = VISUALIZERS.build(pose_config.visualizer)
        visualizer.set_dataset_meta(pose_data_samples[0].dataset_meta)
        for i, (d, f) in enumerate(zip(pose_data_samples, frames_)):
            visualizer.add_datasample(
                'result',
                f,
                data_sample=d,
                draw_gt=False,
                draw_heatmap=False,
                draw_bbox=True,
                show=False,
                wait_time=0,
                out_file=None,
                kpt_thr=0.3)
            frames_[i] = visualizer.get_image()
            cv2.putText(frames_[i], action_result, (10, 30), FONTFACE,
                        FONTSCALE, FONTCOLOR, THICKNESS, LINETYPE)

    for i in range(na):
        anno = annotations[i]
        if anno is None:
            continue
        for j in range(nfpa):
            ind = i * nfpa + j
            frame = frames_[ind]

            # add action result for whole video
            cv2.putText(frame, action_result, (10, 30), FONTFACE, FONTSCALE,
                        FONTCOLOR, THICKNESS, LINETYPE)

            # add spatio-temporal action detection results
            for ann in anno:
                box = ann[0]
                label = ann[1]
                if not len(label):
                    continue
                score = ann[2]
                box = (box * scale_ratio).astype(np.int64)
                st, ed = tuple(box[:2]), tuple(box[2:])
                if not pose_data_samples:
                    cv2.rectangle(frame, st, ed, plate[0], 2)

                for k, lb in enumerate(label):
                    if k >= max_num:
                        break
                    text = abbrev(lb)
                    text = ': '.join([text, f'{score[k]:.3f}'])
                    location = (0 + st[0], 18 + k * 18 + st[1])
                    textsize = cv2.getTextSize(text, FONTFACE, FONTSCALE,
                                               THICKNESS)[0]
                    textwidth = textsize[0]
                    diag0 = (location[0] + textwidth, location[1] - 14)
                    diag1 = (location[0], location[1] + 2)
                    cv2.rectangle(frame, diag0, diag1, plate[k + 1], -1)
                    cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                                FONTCOLOR, THICKNESS, LINETYPE)

    return frames_


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument(
        '--rgb-stdet-config',
        default=(
            'configs/detection/slowonly/'
            'slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.py'
        ),
        help='rgb-based spatio temporal detection config file path')
    parser.add_argument(
        '--rgb-stdet-checkpoint',
        default=('https://download.openmmlab.com/mmaction/detection/ava/'
                 'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/'
                 'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb'
                 '_20201217-16378594.pth'),
        help='rgb-based spatio temporal detection checkpoint file/url')
    parser.add_argument(
        '--skeleton-stdet-checkpoint',
        default=('https://download.openmmlab.com/mmaction/skeleton/posec3d/'
                 'posec3d_ava.pth'),
        help='skeleton-based spatio temporal detection checkpoint file/url')
    parser.add_argument(
        '--det-config',
        default='demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('http://download.openmmlab.com/mmdetection/v2.0/'
                 'faster_rcnn/faster_rcnn_r50_fpn_2x_coco/'
                 'faster_rcnn_r50_fpn_2x_coco_'
                 'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--pose-config',
        default='demo/demo_configs'
        '/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py',
        help='human pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-checkpoint',
        default=('https://download.openmmlab.com/mmpose/top_down/hrnet/'
                 'hrnet_w32_coco_256x192-c78dce93_20200708.pth'),
        help='human pose estimation checkpoint file/url')
    parser.add_argument(
        '--skeleton-config',
        default='configs/skeleton/posec3d'
        '/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py',
        help='skeleton-based action recognition config file path')
    parser.add_argument(
        '--skeleton-checkpoint',
        default='https://download.openmmlab.com/mmaction/skeleton/posec3d/'
        'posec3d_k400.pth',
        help='skeleton-based action recognition checkpoint file/url')
    parser.add_argument(
        '--rgb-config',
        default='configs/recognition/tsn/'
        'tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py',
        help='rgb-based action recognition config file path')
    parser.add_argument(
        '--rgb-checkpoint',
        default='https://download.openmmlab.com/mmaction/recognition/'
        'tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/'
        'tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth',
        help='rgb-based action recognition checkpoint file/url')
    parser.add_argument(
        '--use-skeleton-stdet',
        action='store_true',
        help='use skeleton-based spatio temporal detection method')
    parser.add_argument(
        '--use-skeleton-recog',
        action='store_true',
        help='use skeleton-based action recognition method')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')
    parser.add_argument(
        '--action-score-thr',
        type=float,
        default=0.4,
        help='the threshold of action prediction score')
    parser.add_argument(
        '--video',
        default='demo/test_video_structuralize.mp4',
        help='video file/url')
    parser.add_argument(
        '--label-map-stdet',
        default='tools/data/ava/label_map.txt',
        help='label map file for spatio-temporal action detection')
    parser.add_argument(
        '--label-map',
        default='tools/data/kinetics/label_map_k400.txt',
        help='label map file for action recognition')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--out-filename',
        default='demo/test_stdet_recognition_output.mp4',
        help='output filename')
    parser.add_argument(
        '--predict-stepsize',
        default=8,
        type=int,
        help='give out a spatio-temporal detection prediction per n frames')
    parser.add_argument(
        '--output-stepsize',
        default=1,
        type=int,
        help=('show one frame per n frames in the demo, we should have: '
              'predict_stepsize % output_stepsize == 0'))
    parser.add_argument(
        '--output-fps',
        default=24,
        type=int,
        help='the fps of demo video output')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    args = parser.parse_args()
    return args


def load_label_map(file_path):
    """Load Label Map.

    Args:
        file_path (str): The file path of label map.

    Returns:
        dict: The label map (int -> label name).
    """
    lines = open(file_path).readlines()
    lines = [x.strip().split(': ') for x in lines]
    return {int(x[0]): x[1] for x in lines}


def abbrev(name):
    """Get the abbreviation of label name:

    'take (an object) from (a person)' -> 'take ... from ...'
    """
    while name.find('(') != -1:
        st, ed = name.find('('), name.find(')')
        name = name[:st] + '...' + name[ed + 1:]
    return name


def pack_result(human_detection, result, img_h, img_w):
    """Short summary.

    Args:
        human_detection (np.ndarray): Human detection result.
        result (type): The predicted label of each human proposal.
        img_h (int): The image height.
        img_w (int): The image width.

    Returns:
        tuple: Tuple of human proposal, label name and label score.
    """
    human_detection[:, 0::2] /= img_w
    human_detection[:, 1::2] /= img_h
    results = []
    if result is None:
        return None
    for prop, res in zip(human_detection, result):
        res.sort(key=lambda x: -x[1])
        results.append(
            (prop.data.cpu().numpy(), [x[0] for x in res], [x[1]
                                                            for x in res]))
    return results


def expand_bbox(bbox, h, w, ratio=1.25):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    width = x2 - x1
    height = y2 - y1

    square_l = max(width, height)
    new_width = new_height = square_l * ratio

    new_x1 = max(0, int(center_x - new_width / 2))
    new_x2 = min(int(center_x + new_width / 2), w)
    new_y1 = max(0, int(center_y - new_height / 2))
    new_y2 = min(int(center_y + new_height / 2), h)
    return (new_x1, new_y1, new_x2, new_y2)


def cal_iou(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    intersect = w * h
    union = s1 + s2 - intersect
    iou = intersect / union

    return iou


def skeleton_based_action_recognition(args, pose_results, h, w):
    label_map = [x.strip() for x in open(args.label_map).readlines()]
    num_class = len(label_map)
    #기존 frame-{bbox,keypoint}형태를 crop-frame-{keypoint}형태로 변경
    #frame 마다 돌기
    frame_num = len(pose_results)
    crop_pose_results = []
    bbox_num = len(pose_results[0]['bboxes'])
    for bbox_idx in range(bbox_num):
        person_pose_result = []
        for frame_idx in range(frame_num):
            crop_pose_result = {}
            #keypoint[0]이 y, [1]이 x
            if bbox_idx>len(pose_results[frame_idx]['bboxes'])-1:
                continue
            crop_pose_result['keypoints'] = pose_results[frame_idx]['keypoints'][bbox_idx]
            crop_pose_result['keypoint_scores'] = pose_results[frame_idx]['keypoint_scores'][bbox_idx][np.newaxis]
            crop_pose_result['bbox_scores'] = pose_results[frame_idx]['bbox_scores'][bbox_idx][np.newaxis]
            crop_pose_result['bboxes'] = pose_results[frame_idx]['bboxes'][bbox_idx]
            crop_pose_result['bboxes'][0], crop_pose_result['bboxes'][1], crop_pose_result['bboxes'][2], crop_pose_result['bboxes'][3] = 0,0,w,h
            x1, y1, x2, y2 = pose_results[frame_idx]['bboxes'][bbox_idx][0], pose_results[frame_idx]['bboxes'][bbox_idx][1], pose_results[frame_idx]['bboxes'][bbox_idx][2], pose_results[frame_idx]['bboxes'][bbox_idx][3]
            crop_pose_result['keypoints'][:, 1] = (crop_pose_result['keypoints'][:, 1] - x1) * (w / (x2 - x1))
            crop_pose_result['keypoints'][:, 0] = (crop_pose_result['keypoints'][:, 0] - y1) * (h / (y2 - y1))

            crop_pose_result['bboxes'] = crop_pose_result['bboxes'][np.newaxis]
            crop_pose_result['keypoints'] = crop_pose_result['keypoints'][np.newaxis]
            person_pose_result.append(crop_pose_result)
        crop_pose_results.append(person_pose_result)

        
    skeleton_config = mmengine.Config.fromfile(args.skeleton_config)
    skeleton_config.model.cls_head.num_classes = num_class  # for K400 dataset

    results = []
    skeleton_model = init_recognizer(
        skeleton_config, args.skeleton_checkpoint, device=args.device)
    for crop_pose_result in crop_pose_results:
        result = inference_skeleton(skeleton_model, crop_pose_result, (h, w))
        print(result.get('pred_label'))
        results.append(label_map[result.get('pred_label').item()])

    return results


def skeleton_based_action_recognition_m(args, pose_results, h, w, skeleton_model, modified_results):
    label_map = [x.strip() for x in open(args.label_map).readlines()]
    num_class = len(label_map)
    #기존 frame-{bbox,keypoint}형태를 crop-frame-{keypoint}형태로 변경
    #frame 마다 돌기
    frame_num = len(pose_results)
    crop_pose_results = []
    bbox_num = len(pose_results[0]['bboxes'])

    person_pose_dic = {}
    #person이 나오는 순서가 같다는 가정
    for bbox_idx in range(bbox_num):
        person_pose_result = []        
        for frame_idx in range(frame_num):
            person = 0
            #몇번쨰 사람인지 확인
            id = 0
            for person_id in len(modified_results[frame_idx]):
                if modified_results[frame_idx][person_id][5] == 0.0:
                    if bbox_idx == person:
                        id = modified_results[frame_idx][person_id][0]
                        break
                    else:
                        person += 1


            crop_pose_result = {}
            #keypoint[0]이 y, [1]이 x
            if bbox_idx>len(pose_results[frame_idx]['bboxes'])-1:
                continue
            crop_pose_result['keypoints'] = pose_results[frame_idx]['keypoints'][bbox_idx]
            crop_pose_result['keypoint_scores'] = pose_results[frame_idx]['keypoint_scores'][bbox_idx][np.newaxis]
            crop_pose_result['bbox_scores'] = pose_results[frame_idx]['bbox_scores'][bbox_idx][np.newaxis]
            crop_pose_result['bboxes'] = pose_results[frame_idx]['bboxes'][bbox_idx]
            crop_pose_result['bboxes'][0], crop_pose_result['bboxes'][1], crop_pose_result['bboxes'][2], crop_pose_result['bboxes'][3] = 0,0,w,h
            x1, y1, x2, y2 = pose_results[frame_idx]['bboxes'][bbox_idx][0], pose_results[frame_idx]['bboxes'][bbox_idx][1], pose_results[frame_idx]['bboxes'][bbox_idx][2], pose_results[frame_idx]['bboxes'][bbox_idx][3]
            crop_pose_result['keypoints'][:, 1] = (crop_pose_result['keypoints'][:, 1] - x1) * (w / (x2 - x1))
            crop_pose_result['keypoints'][:, 0] = (crop_pose_result['keypoints'][:, 0] - y1) * (h / (y2 - y1))

            crop_pose_result['bboxes'] = crop_pose_result['bboxes'][np.newaxis]
            crop_pose_result['keypoints'] = crop_pose_result['keypoints'][np.newaxis]
            if id not in person_pose_dic:
                person_pose_dic[id] = []
            person_pose_dic[id].append(crop_pose_result)

        
    results = {}
    for key in person_pose_dic:
        results[key] = []
        result = inference_skeleton(skeleton_model, person_pose_dic[key], (h, w))
        results[key] = label_map[result.get('pred_label').item()]

    return results

def rgb_based_action_recognition(args):
    rgb_config = mmengine.Config.fromfile(args.rgb_config)
    rgb_config.model.backbone.pretrained = None
    rgb_model = init_recognizer(rgb_config, args.rgb_checkpoint, args.device)
    action_results = inference_recognizer(rgb_model, args.video)
    rgb_action_result = action_results.pred_score.argmax().item()
    label_map = [x.strip() for x in open(args.label_map).readlines()]
    return label_map[rgb_action_result]

def crop_and_normalize_images_array(image_array, bounding_boxes, target_size=(256, 340)):
    # Crop된 이미지를 담을 리스트
    crop_frames = []
    for image in image_array:
        cropped_images = []
        for bboxes in bounding_boxes:
            # bbox 좌표 추출
            for bbox in bboxes:
                bbox = bbox.to(dtype=torch.int)
                x1, y1, x2, y2 = bbox[0].item(),bbox[1].item(),bbox[2].item(),bbox[3].item()

                # bbox 좌표로 이미지 crop
                cropped_image = image[y1:y2, x1:x2, :]

                # 이미지 크기를 목표 크기로 조절
                cropped_image = Image.fromarray(cropped_image.astype(np.uint8)).resize(target_size, Image.BILINEAR)
                cropped_image_array = np.array(cropped_image)
                
                # crop된 이미지를 리스트에 추가
                cropped_images.append(cropped_image_array)
        crop_frames.append(cropped_images)

    return np.array(crop_frames)


def crop_and_normalize_images_array_m(image_array, bounding_boxes, target_size=(340,256)):
    #frame 마다
    person_dic = {}
    num = 0

    for bboxes in bounding_boxes:
        # bbox 좌표 추출
        for bbox in bboxes:
            id = bbox[0]
            
            if bbox[5] == 0.0:
                if id not in person_dic:
                    person_dic[id] = []
                bbox = [int(value) for value in bbox]
                x1, y1, x2, y2 = bbox[1],bbox[2],bbox[3],bbox[4]
                x_f = 0
                x_f = 0
                y_f = 0
                y_l = 0
                # bbox 좌표로 이미지 crop
                if x1>x2:
                    x_f = x2
                    x_l = x1
                else:
                    x_f = x1
                    x_l = x2   
                if y1>y2:
                    y_f = y2
                    y_l = y1
                else:
                    y_f = y1
                    y_l = y2     

                if x_f<0:
                    x_f = 0          
                if y_f<0:
                    y_f = 0    
                cropped_image = image_array[num][y_f:y_l, x_f:x_l, :]
                cropped_image_array = None
                # 이미지 크기를 목표 크기로 조절
                cropped_image = Image.fromarray(cropped_image.astype(np.uint8)).resize(target_size, Image.BILINEAR)
                cropped_image_array = np.array(cropped_image)
                print(id, ' ', cropped_image_array.shape, ' ')
                
                # crop된 이미지를 리스트에 추가
                # cropped_images.append(cropped_image_array)
                person_dic[id].append(cropped_image_array)
        num += 1

    # return np.array(crop_frames)
    return person_dic


def skeleton_based_stdet(args, label_map, human_detections, pose_results,
                         num_frame, clip_len, frame_interval, h, w):
    window_size = clip_len * frame_interval
    assert clip_len % 2 == 0, 'We would like to have an even clip_len'
    timestamps = np.arange(window_size // 2, num_frame + 1 - window_size // 2,
                           args.predict_stepsize)

    skeleton_config = mmengine.Config.fromfile(args.skeleton_config)
    num_class = max(label_map.keys()) + 1  # for AVA dataset (81)
    skeleton_config.model.cls_head.num_classes = num_class
    skeleton_stdet_model = init_recognizer(skeleton_config,
                                           args.skeleton_stdet_checkpoint,
                                           args.device)

    skeleton_predictions = []

    print('Performing SpatioTemporal Action Detection for each clip')
    prog_bar = mmengine.ProgressBar(len(timestamps))
    for timestamp in timestamps:
        proposal = human_detections[timestamp - 1]
        if proposal.shape[0] == 0:  # no people detected
            skeleton_predictions.append(None)
            continue

        start_frame = timestamp - (clip_len // 2 - 1) * frame_interval
        frame_inds = start_frame + np.arange(0, window_size, frame_interval)
        frame_inds = list(frame_inds - 1)
        num_frame = len(frame_inds)  # 30

        pose_result = [pose_results[ind] for ind in frame_inds]

        skeleton_prediction = []
        for i in range(proposal.shape[0]):  # num_person
            skeleton_prediction.append([])

            fake_anno = dict(
                frame_dict='',
                label=-1,
                img_shape=(h, w),
                origin_shape=(h, w),
                start_index=0,
                modality='Pose',
                total_frames=num_frame)
            num_person = 1

            num_keypoint = 17
            keypoint = np.zeros(
                (num_person, num_frame, num_keypoint, 2))  # M T V 2
            keypoint_score = np.zeros(
                (num_person, num_frame, num_keypoint))  # M T V

            # pose matching
            person_bbox = proposal[i][:4]
            area = expand_bbox(person_bbox, h, w)

            for j, poses in enumerate(pose_result):  # num_frame
                max_iou = float('-inf')
                index = -1
                if len(poses['keypoints']) == 0:
                    continue
                for k, bbox in enumerate(poses['bboxes']):
                    iou = cal_iou(bbox, area)
                    if max_iou < iou:
                        index = k
                        max_iou = iou
                keypoint[0, j] = poses['keypoints'][index]
                keypoint_score[0, j] = poses['keypoint_scores'][index]

            fake_anno['keypoint'] = keypoint
            fake_anno['keypoint_score'] = keypoint_score

            output = inference_recognizer(skeleton_stdet_model, fake_anno)
            # for multi-label recognition
            score = output.pred_score.tolist()
            for k in range(len(score)):  # 81
                if k not in label_map:
                    continue
                if score[k] > args.action_score_thr:
                    skeleton_prediction[i].append((label_map[k], score[k]))

        skeleton_predictions.append(skeleton_prediction)
        prog_bar.update()

    return timestamps, skeleton_predictions


def rgb_based_stdet(args, frames, label_map, human_detections, w, h, new_w,
                    new_h, w_ratio, h_ratio):

    rgb_stdet_config = mmengine.Config.fromfile(args.rgb_stdet_config)
    rgb_stdet_config.merge_from_dict(args.cfg_options)

    val_pipeline = rgb_stdet_config.val_pipeline
    
    sampler = [x for x in val_pipeline if x['type'] == 'MMUniformSampleFrames'][0]
    # sampler = [x for x in val_pipeline if x['type'] == 'SampleAVAFrames'][0]

    #임의로 추가
    sampler['frame_interval'] = 1
    sampler['clip_len'] = sampler['clip_len']['RGB']

    clip_len, frame_interval = sampler['clip_len'], sampler['frame_interval']
    assert clip_len % 2 == 0, 'We would like to have an even clip_len'

    window_size = clip_len * frame_interval
    num_frame = len(frames)
    # Note that it's 1 based here
    timestamps = np.arange(window_size // 2, num_frame + 1 - window_size // 2,
                           args.predict_stepsize)

    # Get img_norm_cfg
    img_norm_cfg = dict(
        mean=np.array(rgb_stdet_config.model.data_preprocessor.mean),
        std=np.array(rgb_stdet_config.model.data_preprocessor.std),
        to_rgb=False)

    # Build STDET model
    try:
        # In our spatiotemporal detection demo, different actions should have
        # the same number of bboxes.
        rgb_stdet_config['model']['test_cfg']['rcnn'] = dict(action_thr=0)
    except KeyError:
        pass

    rgb_stdet_config.model.backbone.pretrained = None
    rgb_stdet_model = init_detector(
        rgb_stdet_config, args.rgb_stdet_checkpoint, device=args.device)

    predictions = []

    print('Performing SpatioTemporal Action Detection for each clip')
    prog_bar = mmengine.ProgressBar(len(timestamps))
    # for timestamp, proposal in zip(timestamps, human_detections):
    for timestamp in timestamps:
        proposal = human_detections[timestamp - 1]
        if proposal.shape[0] == 0:
            predictions.append(None)
            continue
        
        # #bbox얻기
        # person_bbox = proposal[i][:4]
        # area = expand_bbox(person_bbox, h, w)
        
        start_frame = timestamp - (clip_len // 2 - 1) * frame_interval
        frame_inds = start_frame + np.arange(0, window_size, frame_interval)
        frame_inds = list(frame_inds - 1)

        
        imgs = [frames[ind].astype(np.float32) for ind in frame_inds]

        #shw
        crop_imgs = []
        
        #8frame에 대한 각 객체디텍션 결과
        #TcropHWC
        crop_imgs = crop_and_normalize_images_array(imgs, human_detections[start_frame:start_frame+clip_len])
        #cropTHWC
        temp_crop_imgs = np.stack(crop_imgs).transpose((1, 0, 2, 3, 4))


        _ = [mmcv.imnormalize_(img, **img_norm_cfg) for img in imgs]
        
        #cropTHWC
        result_crop_imgs = []
        for crop_img in temp_crop_imgs:
            temp = []
            for frame_crop_img in crop_img:
                temp.append(frame_crop_img.astype(np.float32))
            _ = [mmcv.imnormalize_(img, **img_norm_cfg) for img in temp]
            result_crop_imgs.append(temp)



        # THWC -> CTHW -> 1CTHW
        # input_array = np.stack(imgs).transpose((3, 0, 1, 2))[np.newaxis]
        # input_tensor = torch.from_numpy(input_array).to(args.device)
        # input_array = np.stack(crop_imgs).transpose((0, 2, 5, 1, 3, 4))[np.newaxis]
        # result_crop_imgs = np.array(result_crop_imgs)
        
        #바운딩 박스마다의 결과
        #dic으로 변경하기
        prediction = []

        #cropTHWC->crop[1CTHW]
        for input in result_crop_imgs:
            input = np.array(input)
            input_array = np.stack(input).transpose((3, 0, 1, 2))[np.newaxis][np.newaxis]
            input_tensor = torch.from_numpy(input_array).to(args.device)

            datasample = ActionDataSample()
            datasample.proposals = InstanceData(bboxes=proposal)
            datasample.set_metainfo(dict(img_shape=(new_h, new_w)))


            with torch.no_grad():
                result = rgb_stdet_model(
                    input_tensor, [datasample], mode='predict')
                scores = result[0].pred_score.data
                label = result[0].pred_label.data
            
            #selfharm에 대한 score가 일정수준 이상일 경우 selfharm으로 판단
            if scores[1] > args.action_score_thr:
                prediction.append(label_map[1])
            else:
                prediction.append(label_map[0])
        predictions.append(prediction)

        prog_bar.update()

    return timestamps, predictions

#for module
#frames는 len 8
def rgb_based_stdet_m(args, frames, label_map, human_detections, w, h, new_w,
                    new_h, w_ratio, h_ratio, rgb_stdet_model, img_norm_cfg, modified_results):

    predictions = []

    print('Performing SpatioTemporal Action Detection for each clip')
    # for timestamp, proposal in zip(timestamps, human_detections):

    proposal = human_detections[0]
    if proposal.shape[0] == 0:
        predictions.append(None)
        return predictions  

    
    imgs = []
    for f in frames:
        img = f.astype(np.float32)
        imgs.append(img)


    #shw
    # crop_imgs = []
    
    #8frame에 대한 각 객체디텍션 결과
    #TcropHWC
    # crop_imgs = crop_and_normalize_images_array_m(imgs, human_detections[0:len(imgs)])
    result_dic = {}
    #cropTHWC
    crop_dic = crop_and_normalize_images_array_m(imgs, modified_results[0:len(imgs)])
    for key in crop_dic:
        temp = np.array(crop_dic[key]).astype(np.float32)
        _ = [mmcv.imnormalize_(img, **img_norm_cfg) for img in temp]

        #cropTHWC->crop[1CTHW]
        input = np.array(temp)
        input_array = np.stack(input).transpose((3, 0, 1, 2))[np.newaxis][np.newaxis]
        input_tensor = torch.from_numpy(input_array).to(args.device)

        datasample = ActionDataSample()
        datasample.proposals = InstanceData(bboxes=proposal)
        datasample.set_metainfo(dict(img_shape=(new_h, new_w)))


        with torch.no_grad():
            result = rgb_stdet_model(
                input_tensor, [datasample], mode='predict')
            scores = result[0].pred_score.data
            label = result[0].pred_label.data
        
        #selfharm에 대한 score가 일정수준 이상일 경우 selfharm으로 판단
        if scores[1] > args.action_score_thr:
            result_dic[key] = label_map[2]
        else:
            result_dic[key] = label_map[1]


    return result_dic

#person_bboxes [id, bbox]
#rgb = 8frame, pose = 30frame
def selfharm_detection():
    args = parse_args()
    root_path = '/selfharm_PLASS/'

    #필요한 루트
    args.video = root_path + 'demo/demo.mp4'
    args.out_filename = root_path + 'demo/result/demo.mp4'

    # args.video = '/workspace/police_lab/mmaction2_mhncity/demo/video/selfharm_day_1018_blue_wsh-25of60.mp4'
    # args.out_filename = '/workspace/police_lab/mmaction2_mhncity/demo/result/selfharm_day_1018_blue_wsh-25of60.mp4'

    args.rgb_stdet_config = root_path + 'work_dirs/demo_rgbposec3d/rgb_only_custom.py'
    args.rgb_stdet_checkpoint = root_path + 'work_dirs/demo_rgbposec3d/rgb_best_acc_top1_epoch_17.pth'
    args.skeleton_config = root_path + 'work_dirs/demo_rgbposec3d/pose_only_custom.py'
    args.skeleton_checkpoint = root_path + 'work_dirs/demo_rgbposec3d/pose_best_acc_top1_epoch_17.pth'
    args.use_skeleton_recog = True

    args.label_map_stdet = root_path + 'demo/label_map_c2_stdet.txt'
    args.label_map = root_path + 'demo/label_map_c2.txt'

    # Load spatio-temporal detection label_map
    stdet_label_map = load_label_map(args.label_map_stdet)
    rgb_stdet_config = mmengine.Config.fromfile(args.rgb_stdet_config)
    rgb_stdet_config.merge_from_dict(args.cfg_options)
    try:
        if rgb_stdet_config['data']['train']['custom_classes'] is not None:
            stdet_label_map = {
                id + 1: stdet_label_map[cls]
                for id, cls in enumerate(rgb_stdet_config['data']['train']
                                         ['custom_classes'])
            }
    except KeyError:
        pass

    action_result = None
    stdet_preds = None
    

    #RGB MODEL INIT
    rgb_stdet_config = mmengine.Config.fromfile(args.rgb_stdet_config)
    rgb_stdet_config.merge_from_dict(args.cfg_options)

    # Get img_norm_cfg
    img_norm_cfg = dict(
        mean=np.array(rgb_stdet_config.model.data_preprocessor.mean),
        std=np.array(rgb_stdet_config.model.data_preprocessor.std),
        to_rgb=False)

    # Build STDET model
    try:
        # In our spatiotemporal detection demo, different actions should have
        # the same number of bboxes.
        rgb_stdet_config['model']['test_cfg']['rcnn'] = dict(action_thr=0)
    except KeyError:
        pass

    rgb_stdet_config.model.backbone.pretrained = None
    rgb_stdet_model = init_detector(
        rgb_stdet_config, args.rgb_stdet_checkpoint, device=args.device)
    
    #POSE MODEL INIT
    label_map = [x.strip() for x in open(args.label_map).readlines()]
    num_class = len(label_map)

    skeleton_config = mmengine.Config.fromfile(args.skeleton_config)
    skeleton_config.model.cls_head.num_classes = num_class  # for selfharm dataset

    skeleton_model = init_recognizer(
        skeleton_config, args.skeleton_checkpoint, device=args.device)
    
    long_result_dic = {}

    while True:
        #이런식으로 호출하면 30frame 관련 데이터를 받을 수 있어야 할 것 같아요.
        #mhncity에서 주신 모듈을 붙여서 pose_result를 받아오는데 환경충돌 오류가 생기는데 문제 해결을 못했습니다.
        modified_results, person_bboxes, pose_results, frames = detect_m()

        h, w, _ = frames[0].shape

        # resize frames to shortside 256
        new_w, new_h = mmcv.rescale_size((w, h), (256, np.Inf))
        frames = [mmcv.imresize(img, (new_w, new_h)) for img in frames]
        w_ratio, h_ratio = new_w / w, new_h / h


        #ACTION DETECTION
        #pose는 30frame 사용
        print('Use skeleton-based recognition')
        #pose데이터에도 id가 필요한데 제가 person_bboxes랑 pose_results 내의 bbox 값을 일일이 확인해서 id를 달아주는 것은 비효율적인 것 같습니다..
        #일단 bbox 순서로 그냥 동일한 id를 가진 사람이라고 치고 진행을 했는데 id를 확인할 방법이 있어야 할 것 같아요..
        pose_result_dic = skeleton_based_action_recognition_m(
            args, pose_results, h, w, skeleton_model, modified_results)
        
        #rgb는 8frame만 사용
        print('Use rgb-based SpatioTemporal Action Detection')
        action_result_dic = rgb_based_stdet_m(args, frames[-8:],
                                                    stdet_label_map,
                                                    person_bboxes, w, h,
                                                    new_w, new_h, w_ratio,
                                                    h_ratio, rgb_stdet_model, img_norm_cfg, modified_results)
        
        for key in pose_result_dic:
            if key not in long_result_dic:
                long_result_dic[key] = ['normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal']
            long_result_dic[key].pop(0)
            long_result_dic[key].append(pose_result_dic[key])
            if 'selfharm' not in long_result_dic[key][-1]:
                long_result_dic[key].pop()
                long_result_dic[key].append(action_result_dic[key])

        for key in long_result_dic:
            selfharm = 0
            for i in range(len(long_result_dic[key])):
                if 'selfharm' in long_result_dic[key][i]:
                    selfharm += 1
            
            if selfharm>=5:
                print('id:', key, ' selfharm')



def main():
    # selfharm_detection()
    args = parse_args()
    root_path = '/selfharm_PLASS/'

    #필요한 루트
    args.video = root_path + 'demo/demo.mp4'
    args.out_filename = root_path + 'demo/result/demo.mp4'

    # args.video = '/workspace/police_lab/mmaction2_mhncity/demo/video/selfharm_day_1018_blue_wsh-25of60.mp4'
    # args.out_filename = '/workspace/police_lab/mmaction2_mhncity/demo/result/selfharm_day_1018_blue_wsh-25of60.mp4'

    args.rgb_stdet_config = root_path + 'work_dirs/demo_rgbposec3d/rgb_only_custom.py'
    args.rgb_stdet_checkpoint = root_path + 'work_dirs/demo_rgbposec3d/rgb_best_acc_top1_epoch_17.pth'
    args.skeleton_config = root_path + 'work_dirs/demo_rgbposec3d/pose_only_custom.py'
    args.skeleton_checkpoint = root_path + 'work_dirs/demo_rgbposec3d/pose_best_acc_top1_epoch_17.pth'
    args.use_skeleton_recog = True
    
    #사람 탐지
    args.det_config = root_path + 'demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py'
    args.det_checkpoint = 'http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'
    
    #스켈레톤 추출
    args.pose_config = root_path + 'demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py'
    args.pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'
    args.use_skeleton_recog = True


    args.label_map_stdet = root_path + 'demo/label_map_c2_stdet.txt'
    args.label_map = root_path + 'demo/label_map_c2.txt'

    args.action_score_thr = 0.7

    tmp_dir = tempfile.TemporaryDirectory()
        
    frame_paths, original_frames = frame_extract(
        args.video, out_dir=tmp_dir.name)
    num_frame = len(frame_paths)
    h, w, _ = original_frames[0].shape

    # Get Human detection results and pose results
    human_detections, _ = detection_inference(
        args.det_config,
        args.det_checkpoint,
        frame_paths,
        args.det_score_thr,
        device=args.device)
    pose_datasample = None
    if args.use_skeleton_recog or args.use_skeleton_stdet:
        pose_results, pose_datasample = pose_inference(
            args.pose_config,
            args.pose_checkpoint,
            frame_paths,
            human_detections,
            device=args.device)

    # resize frames to shortside 256
    new_w, new_h = mmcv.rescale_size((w, h), (256, np.Inf))
    frames = [mmcv.imresize(img, (new_w, new_h)) for img in original_frames]
    w_ratio, h_ratio = new_w / w, new_h / h

    # Load spatio-temporal detection label_map
    stdet_label_map = load_label_map(args.label_map_stdet)
    rgb_stdet_config = mmengine.Config.fromfile(args.rgb_stdet_config)
    rgb_stdet_config.merge_from_dict(args.cfg_options)
    try:
        if rgb_stdet_config['data']['train']['custom_classes'] is not None:
            stdet_label_map = {
                id + 1: stdet_label_map[cls]
                for id, cls in enumerate(rgb_stdet_config['data']['train']
                                         ['custom_classes'])
            }
    except KeyError:
        pass

    action_result = None
    if args.use_skeleton_recog:
        print('Use skeleton-based recognition')
        action_result = skeleton_based_action_recognition(
            args, pose_results, h, w)
    else:
        print('Use rgb-based recognition')
        action_result = rgb_based_action_recognition(args)

    stdet_preds = None
    if args.use_skeleton_stdet:
        print('Use skeleton-based SpatioTemporal Action Detection')
        clip_len, frame_interval = 30, 1
        timestamps, stdet_preds = skeleton_based_stdet(args, stdet_label_map,
                                                       human_detections,
                                                       pose_results, num_frame,
                                                       clip_len,
                                                       frame_interval, h, w)
        for i in range(len(human_detections)):
            det = human_detections[i]
            det[:, 0:4:2] *= w_ratio
            det[:, 1:4:2] *= h_ratio
            human_detections[i] = torch.from_numpy(det[:, :4]).to(args.device)

    else:
        print('Use rgb-based SpatioTemporal Action Detection')
        for i in range(len(human_detections)):
            det = human_detections[i]
            det[:, 0:4:2] *= w_ratio
            det[:, 1:4:2] *= h_ratio
            human_detections[i] = torch.from_numpy(det[:, :4]).to(args.device)
        timestamps, stdet_preds = rgb_based_stdet(args, frames,
                                                  stdet_label_map,
                                                  human_detections, w, h,
                                                  new_w, new_h, w_ratio,
                                                  h_ratio)

    # stdet_results = []
    # for timestamp, prediction in zip(timestamps, stdet_preds):
    #     human_detection = human_detections[timestamp - 1]
    #     stdet_results.append(
    #         pack_result(human_detection, prediction, new_h, new_w))

    # def dense_timestamps(timestamps, n):
    #     """Make it nx frames."""
    #     old_frame_interval = (timestamps[1] - timestamps[0])
    #     start = timestamps[0] - old_frame_interval / n * (n - 1) / 2
    #     new_frame_inds = np.arange(
    #         len(timestamps) * n) * old_frame_interval / n + start
    #     return new_frame_inds.astype(np.int64)

    # dense_n = int(args.predict_stepsize / args.output_stepsize)
    # output_timestamps = dense_timestamps(timestamps, dense_n)
    # frames = [
    #     cv2.imread(frame_paths[timestamp - 1])
    #     for timestamp in output_timestamps
    # ]

    # if args.use_skeleton_recog or args.use_skeleton_stdet:
    #     pose_datasample = [
    #         pose_datasample[timestamp - 1] for timestamp in output_timestamps
    #     ]

    # vis_frames = visualize(args, frames, stdet_results, pose_datasample,
    #                        action_result)
    # vid = mpy.ImageSequenceClip(vis_frames, fps=args.output_fps)
    # vid.write_videofile(args.out_filename)

    tmp_dir.cleanup()


if __name__ == '__main__':
    main()
