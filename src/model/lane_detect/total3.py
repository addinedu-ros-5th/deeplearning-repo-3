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
from collections import deque, Counter

class TrafficLightDetector:
    def __init__(self, model_path, history_length=30):
        self.model = YOLO(model_path)
        self.desired_width = 640
        self.desired_height = 640
        self.history_length = history_length
        self.size_history = deque(maxlen=history_length)
        self.signal_detected = False
        self.signal_color = None
        self.classes = {
            0: 'Red',
            1: 'Red and Green Arrow',
            2: 'Green',
            3: 'Yellow',
            4: 'Green and Green Arrow',
            5: 'Green Arrow'
        }

    def detect_traffic_light(self, frame):
        frame = cv2.resize(frame, (self.desired_width, self.desired_height))
        results = self.model.track(frame, persist=True, verbose=False)
        detected_objects = []

        if results[0].boxes.is_track:
            self.signal_detected = True
            for detection in results[0].boxes.data:
                x1, y1, x2, y2 = detection[:4]
                size = (x2 - x1) * (y2 - y1)  # 신호등 크기 계산
                detected_objects.append(size)
                class_id = int(detection[6])
                if class_id == 0:
                    self.signal_color = "Red"
                elif class_id == 1:
                    self.signal_color = "Red and Green Arrow"
                elif class_id == 2:
                    self.signal_color = "Green"
                elif class_id == 3:
                    self.signal_color = "Yellow"
                elif class_id == 4:
                    self.signal_color = "Green and Green Arrow"
                elif class_id == 5:
                    self.signal_color = "Green Arrow"

        else:
            self.signal_detected = False

        # 업데이트된 크기 히스토리
        self.size_history.extend(detected_objects)
        
        # 신호등이 인식되었을 때'만' 주행 상태 판단
        if self.signal_detected:
            is_driving = self.check_driving_status()
            if is_driving and self.signal_color in ["Red", "Red and Green Arrow"]:
                Pass_Fail = "Fail"
            elif is_driving and self.signal_color in ["Green", "Yellow", "Green and Green Arrow", "Green Arrow"]:
                Pass_Fail = "Pass"
            elif not is_driving and self.signal_color in ["Red", "Red and Green Arrow"]:
                Pass_Fail = "Pass"
            elif not is_driving and self.signal_color in ["Green", "Yellow", "Green and Green Arrow", "Green Arrow"]:
                Pass_Fail = "Fail"
            return is_driving, results, Pass_Fail
        
        else:
            return None, results, None

    def check_driving_status(self): #주행 == true 주행  == false
        # 신호등 크기의 변화를 확인하여 주행 상태 판단
        if len(self.size_history) == self.history_length:
            # 최근 크기 변화 계산
            recent_changes = max(self.size_history) - min(self.size_history)
            # print(f"recent_changes: {recent_changes.item()}")
            # 주행 상태 판단: 크기 변화가 있는 경우 주행 중으로 판단
            if recent_changes > threshold:
                return True
        # 주행 상태 판단: 크기 변화가 없는 경우 정지 중으로 판단
        return False

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    threshold = 10
    person_size_threshold = 5000  # 사람 객체 크기 임계값 설정
    car_size_threshold = 10000    # 자동차 객체 크기 임계값 설정
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
    # yolo_model = YOLO('/home/addinedu/dev_ws/src/ai_project/deeplearning-repo-3/src/model/yolo_detection/all_best.pt')

    # 비디오 파일 로드
    video_path = "/home/addinedu/lane_detect/Ultra-Fast-Lane-Detection/2.MOV"
    cap = cv2.VideoCapture(video_path)

    # 결과 비디오 파일 설정
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    output_path = 'output3.avi'
    img_w, img_h = 1280, 720  # CULane과 Tusimple의 해상도 설정

    if cfg.dataset == 'CULane':
        row_anchor = culane_row_anchor
    elif cfg.dataset == 'Tusimple':
        row_anchor = tusimple_row_anchor
    else:
        raise NotImplementedError

    vout = cv2.VideoWriter(output_path, fourcc, 30.0, (img_w, img_h))

    # 신호등 감지 모델 초기화
    traffic_light_model_path = "/home/addinedu/dev_ws/src/ai_project/deeplearning-repo-3/src/model/traffic_light/traffic_best_ver2.pt"
    traffic_detector = TrafficLightDetector(traffic_light_model_path)

    person_detected_frame_count = 0
    person_detected = False

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
        # cv2.line(frame, (center_x, 0), (center_x, img_h), (255, 0, 0), 2)
        # cv2.line(frame, (center_x2, 0), (center_x2, img_h), (255, 0, 0), 2)

        is_person_detected = False

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

                        if box_size > 80000:
                            if is_driving is not None:
                                pass_fail = "Pass" if not is_driving else "Fail"
                                # cv2.putText(frame, f"Pass/Fail: {pass_fail}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # 신호등 감지
        is_driving, traffic_results, pass_fail = traffic_detector.detect_traffic_light(frame)
        
        # 사람 객체가 감지되면 멈춰야 함
        if is_person_detected:
            if person_detected_frame_count is not None:
                person_detected_frame_count += 1
                if person_detected_frame_count > 50:  # 60프레임 이내에 정지하지 않으면 Fail
                    pass_fail = "Fail"
                else:
                    pass_fail = "Pass" if not is_driving else "Fail"
            # is_driving = False
            
        elif is_driving is not None:
            if is_driving:
                status = "DRIVING"
            else:
                status = "STOP"
            # 결과 출력
            cv2.putText(frame, f"Status: {status}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            if traffic_detector.signal_color is not None:
                # 신호등 색상 출력
                cv2.putText(frame, f"Signal color: {traffic_detector.signal_color}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Pass/Fail: {pass_fail}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        elif not traffic_detector.signal_detected:
            # 신호등이 인식되지 않으면 주행 상태 표시를 제거
            pass

        # 결과 프레임 저장 및 표시
        vout.write(frame)
        cv2.imshow('Lane and Object Detection', frame)

        # 키 입력 대기, 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    vout.release()
    cv2.destroyAllWindows()
    print(f'Results saved to {output_path}')
