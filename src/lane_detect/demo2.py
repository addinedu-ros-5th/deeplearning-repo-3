import torch
import os
import cv2
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import scipy.special
import tqdm
import numpy as np
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset
from data.constant import culane_row_anchor, tusimple_row_anchor
from PIL import Image

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    dist_print('start testing...')
    assert cfg.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    net = parsingNet(pretrained=False, backbone=cfg.backbone,
                     cls_dim=(cfg.griding_num+1, cls_num_per_lane, 4),
                     use_aux=False).cuda()  # we don't need auxiliary segmentation in testing

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # 비디오 파일 로드
    video_path = '/home/addinedu/drive.webm'
    cap = cv2.VideoCapture(video_path)

    # 결과 비디오 파일 설정
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    output_path = 'output.avi'
    img_w, img_h = 1280, 720  # CULane과 Tusimple의 해상도 설정

    if cfg.dataset == 'CULane':
        row_anchor = culane_row_anchor
    elif cfg.dataset == 'Tusimple':
        row_anchor = tusimple_row_anchor
    else:
        raise NotImplementedError

    vout = cv2.VideoWriter(output_path, fourcc, 30.0, (img_w, img_h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 전처리
        input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_frame = Image.fromarray(input_frame)  # numpy 배열을 PIL 이미지로 변환
        input_frame = img_transforms(input_frame)
        input_frame = input_frame.unsqueeze(0).cuda()

        # 모델에 입력
        with torch.no_grad():
            out = net(input_frame)

        col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]

        out_j = out[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(cfg.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == cfg.griding_num] = 0
        out_j = loc

        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1,
                               int(img_h * (row_anchor[cls_num_per_lane - 1 - k] / 288)) - 1)
                        cv2.circle(frame, ppp, 5, (0, 255, 0), -1)

        vout.write(frame)

    cap.release()
    vout.release()
    cv2.destroyAllWindows()
