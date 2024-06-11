import torch
import os
import cv2
import sys
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import scipy.special
import numpy as np
import torchvision.transforms as transforms
from data.constant import culane_row_anchor, tusimple_row_anchor
from PIL import Image
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
from collections import deque, defaultdict
from statistics import mode
import json

class TrafficLightDetector:
    def __init__(self, model_path, history_length=30):
        self.model = YOLO(model_path)
        self.desired_width = 640
        self.desired_height = 640
        self.history_length = history_length
        self.size_history = defaultdict(lambda: deque(maxlen=history_length))
        self.pass_fail_status_history = defaultdict(lambda: deque(maxlen=10))  # pass/fail 상태 기록
        self.pass_fail_status = defaultdict(lambda: "Calculating")  # 초기값을 "Calculating"으로 설정
        self.signal_detected = False
        self.signal_color = defaultdict(lambda: None)
        self.classes = {
            0: 'Red',
            1: 'Red and Green Arrow',
            2: 'Green',
            3: 'Yellow',
            4: 'Green and Green Arrow',
            5: 'Green Arrow'
        }
        self.detected_ids = set()  # 추적된 신호등 ID를 저장하는 집합

    def detect_traffic_light(self, frame):
        frame = cv2.resize(frame, (self.desired_width, self.desired_height))
        results = self.model.track(frame, persist=True)
        detected_objects = []
        
        if results[0].boxes.is_track:
            self.signal_detected = True
            max_size = 0
            max_obj_id = None
            for detection in results[0].boxes.data:
                x1, y1, x2, y2, obj_id, _, class_id = map(int, detection[:7])
                size = (x2 - x1) * (y2 - y1)  # 신호등 크기 계산
                # 가장 크기가 큰 객체만 선택
                if size > max_size:
                    max_size = size
                    max_obj_id = obj_id

            if max_obj_id is not None:
                detected_objects.append((max_obj_id, max_size))
                self.detected_ids.add(max_obj_id)  # 감지된 신호등 ID 추가
                self.signal_color[max_obj_id] = self.classes.get(class_id, None)

        else:
            self.signal_detected = False

        return detected_objects, results

def process_frame(frame, net, yolo_model, img_transforms, row_anchor, cls_num_per_lane, col_sample, col_sample_w, img_w, img_h, prvs, hsv, prev_car_box_sizes):
    next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_frame = Image.fromarray(input_frame)  # numpy 배열을 PIL 이미지로 변환
    input_frame = img_transforms(input_frame)
    input_frame = input_frame.unsqueeze(0).cuda()

    # Optical Flow 계산
    flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Optical Flow의 평균 이동 크기를 계산하여 차량의 움직임 상태 결정
    mean_flow = np.mean(mag)
    movement_status = "Stopped" if mean_flow < 1 else "DRIVING"

    # 모델에 입력
    with torch.no_grad():
        out = net(input_frame)

    out_j = out[0].data.cpu().numpy()
    out_j = out_j[:, ::-1, :]
    prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
    idx = np.arange(cfg.griding_num) + 1
    idx = idx.reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)
    out_j = np.argmax(out_j, axis=0)
    loc[out_j == cfg.griding_num] = 0
    out_j = loc

    lane_areas = []

    # 모든 차선 감지 및 시각화
    for lane_index in range(out_j.shape[1]):
        lane_points = []  # 각 차선의 점들을 저장
        if np.sum(out_j[:, lane_index] != 0) > 2:
            for k in range(out_j.shape[0]):
                if out_j[k, lane_index] > 0:
                    ppp = (int(out_j[k, lane_index] * col_sample_w * img_w / 800) - 1,
                           int(img_h * (row_anchor[cls_num_per_lane - 1 - k] / 288)) - 1)
                    lane_points.append(ppp)  # 점 추가
                    cv2.circle(frame, ppp, 5, (0, 255, 0), -1)

            if len(lane_points) > 1:
                lane_areas.append(np.array(lane_points))

    # YOLO를 사용하여 사람 및 자동차 감지
    yolo_results = yolo_model(source=frame, verbose=False)

    center_x = img_w // 2 - 100
    center_x2 = img_w // 2 + 30

    is_person_detected = False
    car_boxes = []

    for result in yolo_results:
        for box in result.boxes:
            class_ids = int(box.cls)
            # 객체의 바운딩 박스 중심 좌표 계산
            x_center = (box.xyxy[0][0] + box.xyxy[0][2]) // 2
            y_center = (box.xyxy[0][1] + box.xyxy[0][3]) // 2
            # 세로선을 통과하는지 확인
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_size = (x2 - x1) * (y2 - y1)
            if center_x2 > x_center > center_x:
                # 세로선을 통과하는 객체에 대해 바운딩 박스 그리기
                if class_ids == 2 and 2400 < box_size < 15000:  # 자동차 클래스 필터링
                    confidence = box.conf.item()  # 텐서를 스칼라 값으로 변환
                    label = f'Car: {confidence:.2f}'
                    # print(box_size)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if center_x2 > x_center > center_x - 200:
                if class_ids == 0 and 2000 < box_size:  # 사람 클래스 필터링
                    confidence = box.conf.item()  # 텐서를 스칼라 값으로 변환
                    label = f'Person: {confidence:.2f}'
                    # print(box_size)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    is_person_detected = True
            if center_x2 + 70 > x_center > center_x - 70:
                if class_ids == 2 and 20000 < box_size < 100000:  # 자동차 클래스 필터링
                    confidence = box.conf.item()  # 텐서를 스칼라 값으로 변환
                    label = f'Car: {confidence:.2f}'
                    # print(box_size)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    car_boxes.append(box_size)

    # 차량 상태 시각화
    cv2.putText(frame, f'Car status: {movement_status}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame, next_frame, is_person_detected, movement_status, car_boxes

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    # 직접 설정할 인수와 구성 파일 경로 및 모델 파일 경로
    sys.argv = [
        'detection_logic.py',
        './configs/culane.py',
        '--test_model',
        './logs/20240604_014750_lr_1e-01_b_32training_run_1/ep009.pth'
    ]

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

    # YOLOv8 모델 로드
    yolo_model = YOLO('yolov8n.pt')

    # 신호등 감지기 초기화
    traffic_light_model_path = "/home/addinedu/dev_ws/src/ai_project/deeplearning-repo-3/src/model/traffic_light/traffic_best_ver2.pt"
    traffic_light_detector = TrafficLightDetector(traffic_light_model_path)

    # 비디오 파일 로드
    video_path = '/home/addinedu/optical_flow/speed-estimation-of-car-with-optical-flow/data/2.MOV'
    cap = cv2.VideoCapture(video_path)

    # 결과 비디오 파일 설정
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    output_path = 'output5.avi'
    img_w, img_h = 1280, 720  # CULane과 Tusimple의 해상도 설정
    col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
    col_sample_w = col_sample[1] - col_sample[0]

    if cfg.dataset == 'CULane':
        row_anchor = culane_row_anchor
    elif cfg.dataset == 'Tusimple':
        row_anchor = tusimple_row_anchor
    else:
        raise NotImplementedError

    vout = cv2.VideoWriter(output_path, fourcc, 30.0, (img_w, img_h))

    # Optical Flow 초기화
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    frame_counter = 0
    person_detected_frame = -1
    prev_car_box_sizes = []

    log_data = []
    pass_fail_history_person = deque(maxlen=60)
    pass_fail_history_car = deque(maxlen=60)
    pass_fail_history_traffic_light = defaultdict(lambda: deque(maxlen=60))

    with ThreadPoolExecutor(max_workers=4) as executor:
        while cap.isOpened():
            ret, frame2 = cap.read()
            if not ret:
                break

            future = executor.submit(process_frame, frame2, net, yolo_model, img_transforms, row_anchor, cls_num_per_lane, col_sample, col_sample_w, img_w, img_h, prvs, hsv, prev_car_box_sizes)
            frame2, prvs, is_person_detected, movement_status, car_boxes = future.result()

            detected_objects, traffic_light_results = traffic_light_detector.detect_traffic_light(frame2)
            for obj_id, size in detected_objects:
                signal_color = traffic_light_detector.signal_color[obj_id]
                if signal_color is not None:
                    if (signal_color == 'Red' and movement_status == "DRIVING") or \
                       (signal_color in ['Green', 'Green and Green Arrow', 'Green Arrow'] and movement_status == "Stopped"):
                        pass_fail = "fail"
                    else:
                        pass_fail = "pass"
                    cv2.putText(frame2, f"Signal color: {signal_color}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame2, f"Pass/Fail: {pass_fail}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    pass_fail_history_traffic_light[obj_id].append(pass_fail)
                    if len(pass_fail_history_traffic_light[obj_id]) == 30:
                        pass_fail_final = max(set(pass_fail_history_traffic_light[obj_id]), key=pass_fail_history_traffic_light[obj_id].count)
                        log_data.append({
                            'frame': frame_counter,
                            'signal_color': signal_color,
                            'movement_status': movement_status,
                            'pass_fail': pass_fail_final
                        })
                        pass_fail_history_traffic_light[obj_id].clear()

            if is_person_detected:
                person_detected_frame = frame_counter

            if person_detected_frame != -1 and frame_counter - person_detected_frame <= 30:
                if movement_status == "DRIVING":
                    result = "fail"
                else:
                    result = "pass"
                cv2.putText(frame2, f'Result: {result}', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                pass_fail_history_person.append(result)
                if len(pass_fail_history_person) == 60:
                    pass_fail_final = max(set(pass_fail_history_person), key=pass_fail_history_person.count)
                    log_data.append({
                        'frame': frame_counter,
                        'person_detected': True,
                        'movement_status': movement_status,
                        'pass_fail': pass_fail_final
                    })
                    pass_fail_history_person.clear()

            if prev_car_box_sizes:
                for i, car_box in enumerate(car_boxes):
                    if car_box >= 70000 and abs(car_box - prev_car_box_sizes[i]) <= 15000:
                        if movement_status == "DRIVING":
                            result = "fail"
                        else:
                            result = "pass"
                        cv2.putText(frame2, f'Result: {result}', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        pass_fail_history_car.append(result)
                        if len(pass_fail_history_car) == 60:
                            pass_fail_final = max(set(pass_fail_history_car), key=pass_fail_history_car.count)
                            log_data.append({
                                'frame': frame_counter,
                                'car_box_size': car_box,
                                'movement_status': movement_status,
                                'pass_fail': pass_fail_final
                            })
                            pass_fail_history_car.clear()

            prev_car_box_sizes = car_boxes
            frame_counter += 1
            vout.write(frame2)
            cv2.imshow('Lane and Person Detection', frame2)

            # 키 입력 대기, 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    vout.release()
    cv2.destroyAllWindows()
    print(f'Results saved to {output_path}')

    # 로그 데이터를 JSON 파일로 저장
    with open('log_data.json', 'w') as f:
        json.dump(log_data, f, indent=4)
    print(f'Log data saved to log_data.json')

