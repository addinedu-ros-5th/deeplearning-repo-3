import torch
import os
import cv2
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import scipy.special
import numpy as np
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset
from data.constant import culane_row_anchor, tusimple_row_anchor
from PIL import Image
import time

def calculate_perspective_transform(src_points, dst_points):
    return cv2.getPerspectiveTransform(src_points, dst_points)

def calculate_speed(prev_points, current_points, time_elapsed):
    speeds = []
    for (x1, y1), (x2, y2) in zip(prev_points, current_points):
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        speed = distance / time_elapsed  # 속도를 픽셀/초 단위로 계산
        speeds.append(speed)
    if speeds:
        avg_speed = np.mean(speeds)  # 평균 속도 계산
        return avg_speed
    return 0

def pixels_to_kmh(pixels_per_second, pixel_to_meter_ratio):
    # 픽셀/초 단위의 속도를 km/h 단위로 변환
    meters_per_second = pixels_per_second * pixel_to_meter_ratio
    km_per_hour = meters_per_second * 3.6
    return km_per_hour

def draw_projected_grid(frame, points, grid_size, perspective_transform):
    h, w = frame.shape[:2]
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    
    min_x = max(min_x - grid_size, 0)
    min_y = max(min_y - grid_size, 0)
    max_x = min(max_x + grid_size, w)
    max_y = min(max_y + grid_size, h)
    
    for x in range(min_x, max_x, grid_size):
        pt1 = np.array([[x, min_y]], dtype=np.float32)
        pt2 = np.array([[x, max_y]], dtype=np.float32)
        pt1_transformed = cv2.perspectiveTransform(pt1[None, :, :], perspective_transform)[0, 0]
        pt2_transformed = cv2.perspectiveTransform(pt2[None, :, :], perspective_transform)[0, 0]
        cv2.line(frame, tuple(map(int, pt1_transformed)), tuple(map(int, pt2_transformed)), (255, 255, 255), 1)
    for y in range(min_y, max_y, grid_size):
        pt1 = np.array([[min_x, y]], dtype=np.float32)
        pt2 = np.array([[max_x, y]], dtype=np.float32)
        pt1_transformed = cv2.perspectiveTransform(pt1[None, :, :], perspective_transform)[0, 0]
        pt2_transformed = cv2.perspectiveTransform(pt2[None, :, :], perspective_transform)[0, 0]
        cv2.line(frame, tuple(map(int, pt1_transformed)), tuple(map(int, pt2_transformed)), (255, 255, 255), 1)

def calculate_perspective_transform(src_points, dst_points):
    return cv2.getPerspectiveTransform(src_points, dst_points)

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

    video_path = '/home/addinedu/drive.webm'
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    output_path = 'output.avi'
    img_w, img_h = 1280, 720

    if cfg.dataset == 'CULane':
        row_anchor = culane_row_anchor
    elif cfg.dataset == 'Tusimple':
        row_anchor = tusimple_row_anchor
    else:
        raise NotImplementedError

    vout = cv2.VideoWriter(output_path, fourcc, 30.0, (img_w, img_h))
    prev_points = None
    prev_time = time.time()

    # 픽셀을 미터로 변환하는 비율 (예를 들어, 1 픽셀이 0.05 미터에 해당한다고 가정)
    pixel_to_meter_ratio = 0.05

    # 1초 동안의 속도를 저장할 리스트 및 프레임 카운터 초기화
    speeds = []
    frame_counter = 0
    avg_speed_kmh = 0

    grid_size = 50  # 그리드 크기 설정

    # 원근 변환 설정 (예시 좌표, 사용자가 조정 필요)
    src_points = np.float32([[0, 288], [800, 288], [0, 0], [800, 0]])
    dst_points = np.float32([[200, img_h], [img_w - 200, img_h], [200, 0], [img_w - 200, 0]])
    perspective_transform = calculate_perspective_transform(src_points, dst_points)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_frame = Image.fromarray(input_frame)
        input_frame = img_transforms(input_frame)
        input_frame = input_frame.unsqueeze(0).cuda()

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

        current_points = []
        feature_points = []
        # 현재 차선과 마주보고 있는 차선 인덱스 설정 (예시로 idx=2와 idx=3을 사용)
        lane_indices = [2, 1]
        for lane_index in lane_indices:
            if np.sum(out_j[:, lane_index] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, lane_index] > 0:
                        ppp = (int(out_j[k, lane_index] * col_sample_w * img_w / 800) - 1,
                               int(img_h * (row_anchor[cls_num_per_lane - 1 - k] / 288)) - 1)
                        # 모든 점을 녹색으로 표시
                        cv2.circle(frame, ppp, 5, (0, 255, 0), -1)
                        current_points.append(ppp)
                        # 특징 점을 선택 (하단 절반에서 여러 개)
                        if k < out_j.shape[0] // 2:
                            feature_points.append(ppp)

        if prev_points is not None:
            current_time = time.time()
            time_elapsed = current_time - prev_time
            frame_counter += 1
            speed_pixels_per_sec = calculate_speed(prev_points, feature_points, time_elapsed)
            speeds.append(pixels_to_kmh(speed_pixels_per_sec, pixel_to_meter_ratio))
            prev_time = current_time

            # 1초 (30 프레임)마다 평균 속도 계산 및 초기화
            if frame_counter >= 30:
                avg_speed_kmh = np.mean(speeds)
                speeds = []
                frame_counter = 0

        # 특징 점을 파란색으로 표시
        for point in feature_points:
            cv2.circle(frame, point, 5, (255, 0, 0), -1)

        prev_points = feature_points

        # 인식된 도로 위의 점들을 기준으로 격자 그리드 그리기
        if current_points:
            draw_projected_grid(frame, np.array(current_points), grid_size, perspective_transform)

        # 현재 1초 동안의 평균 속도를 표시
        cv2.putText(frame, f"Speed: {avg_speed_kmh:.2f} km/h", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        vout.write(frame)
        cv2.imshow('Lane Detection', frame)

        # 키 입력 대기, 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    vout.release()
    cv2.destroyAllWindows()
