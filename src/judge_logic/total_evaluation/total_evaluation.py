import json
import cv2
import pandas as pd
from collections import defaultdict, deque, Counter
from statistics import mode
from ultralytics import YOLO
import os

class VehicleDetector:
    def __init__(self, model_path, min_vehicle_size=1000):
        self.model_path = model_path
        self.min_vehicle_size = min_vehicle_size
        self.model = YOLO(model_path)
        self.img_width = 1280  # 영상의 가로 해상도
        self.img_height = 720  # 영상의 세로 해상도
        self.center_line_x = self.img_width // 2  # 영상의 중앙선 x 좌표
        self.center_line_tolerance = 70  # 중앙선 허용 오차
        self.last_bbox_size = 0
        self.classes = {
            0: "pedestrian",
            2: "car",
            5: "bus",
            7: "truck",
            # Add more classes if needed
        }
        self.pass_fail_status_history = defaultdict(lambda: deque(maxlen=15))  # 최대 10개 항목 저장
        self.pass_fail_status = {}

    def check_front_vehicle(self, frame):
        frame = cv2.resize(frame, (self.img_width, self.img_height))
        results = self.model.track(frame, persist=True)

        is_crash = 0  # 초기 값 설정
        x1, y1, x2, y2, track_id, class_id = 0, 0, 0, 0, 0, 0  # 초기 값 설정

        for box in results[0].boxes:
            # 바운딩 박스의 중심 좌표 계산
            x_center = (box.xyxy[0][0] + box.xyxy[0][2]) // 2
            y_center = (box.xyxy[0][1] + box.xyxy[0][3]) // 2
            # 중앙선을 통과하는지 확인
            if self.center_line_x - self.center_line_tolerance < x_center < self.center_line_x + self.center_line_tolerance:
                # 바운딩 박스의 크기 계산
                box_size = box.xywh[0][2] * box.xywh[0][3]
                # 자동차 클래스 필터링 및 크기 조건 검사
                if 20000 < box_size < 80000:
                    is_crash = 1  # 전방 차량이 감지됨
                elif box_size > 80000:
                    is_crash = 2  # 전방 차량이 감지됨 충돌 직전
                    if box.id is not None:  # None이 아닌 경우에만 track_id 설정
                        track_id = int(box.id)  # 클래스 ID 추출
                        class_id = int(box.cls)
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                break  # 첫 번째 바운딩 박스만 고려

        return is_crash, x1, y1, x2, y2, track_id, class_id

    def determine_driving_status(self, current_bbox_size):
        # 이전 프레임과의 바운딩 박스 크기 비교
        size_difference = abs(current_bbox_size - self.last_bbox_size)
        # 크기 변화량을 기준으로 주행 상태 판단
        if size_difference > 1000:  # 예시로 1000을 기준으로 주행 중으로 판단
            return True  # 주행 중
        else:
            return False  # 정지 중

    def update_last_bbox_size(self, current_bbox_size):
        self.last_bbox_size = current_bbox_size

    def determine_pass_fail_status(self, is_driving):
        return 'Fail' if is_driving else 'Pass'

class TrafficLightDetector:
    def __init__(self, model_path, history_length=30):
        self.model = YOLO(model_path)
        self.desired_width = 1280
        self.desired_height = 720
        self.history_length = history_length
        self.size_history = defaultdict(lambda: deque(maxlen=history_length))
        self.pass_fail_status_history = defaultdict(lambda: deque(maxlen=15))  # pass/fail 상태 기록
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

                # 업데이트된 크기 히스토리
                self.size_history[max_obj_id].append(max_size)

                # 객체가 발견된 경우에만 Pass/Fail 상태 업데이트
                is_driving = self.check_driving_status(max_obj_id)
                pass_fail = self.determine_pass_fail_status(is_driving, class_id)
                self.pass_fail_status_history[max_obj_id].append(pass_fail)  # 최근 20프레임의 Pass/Fail 기록

                # 객체의 Pass/Fail 값을 최근 10프레임 동안의 데이터를 기반으로 판단
                if len(self.pass_fail_status_history[max_obj_id]) == 15:
                    self.pass_fail_status[max_obj_id] = mode(self.pass_fail_status_history[max_obj_id])

        else:
            self.signal_detected = False

        return detected_objects, results


    def check_driving_status(self, obj_id):
        # 신호등 크기의 변화를 확인하여 주행 상태 판단
        if len(self.size_history[obj_id]) == self.history_length:
            # 최근 크기 변화 계산
            recent_changes = max(self.size_history[obj_id]) - min(self.size_history[obj_id])
            # 주행 상태 판단: 크기 변화가 있는 경우 주행 중으로 판단
            if recent_changes > threshold:
                return True
        # 주행 상태 판단: 크기 변화가 없는 경우 정지 중으로 판단
        return False

    def determine_pass_fail_status(self, is_driving, class_id):
        if is_driving and class_id in [0, 1]:
            Pass_Fail = "Fail"
        elif is_driving and class_id in [2, 3, 4, 5]:
            Pass_Fail = "Pass"
        elif not is_driving and class_id in [0, 1]:
            Pass_Fail = "Pass"
        elif not is_driving and class_id in [2, 3, 4, 5]:
            Pass_Fail = "Fail"
        return Pass_Fail
    
def save_results_to_json(vehicle_results, traffic_light_results, file_path, video_length):
    results = {
        "vehicle_results": vehicle_results,
        "traffic_light_results": traffic_light_results,
        "video_length": video_length
    }
    with open(file_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

def get_most_common_row(group):
    return Counter(map(tuple, group.values)).most_common(1)[0][0]

def format_time(milliseconds):
    seconds = milliseconds // 1000
    minutes = seconds // 60
    seconds %= 60
    formatted_time = f"{minutes:02}m:{seconds:02}s"
    return formatted_time

def main(video_path, vehicle_model_path, traffic_light_model_path):
    cap = cv2.VideoCapture(video_path)
    vehicle_detector = VehicleDetector(vehicle_model_path)
    traffic_light_detector = TrafficLightDetector(traffic_light_model_path)
    df_data = []
    vehicle_results = []
    traffic_light_results = []
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 인풋 영상 파일명에서 확장자 제거
    base_filename = os.path.splitext(os.path.basename(video_path))[0]
    output_filename = f"./data/output_data/output_{base_filename}.mp4"

    # VideoWriter 객체 생성
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정 (예: mp4v)
    out = cv2.VideoWriter(output_filename, fourcc, 30.0, (1280, 720))

    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1

        

        # 차량 감지
        vehicle_is_crash, vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2, vehicle_track_id, vehicle_class_id = vehicle_detector.check_front_vehicle(frame)
        
        # 차량 바운딩 박스 색상 설정
        vehicle_color = (0, 255, 0)  # 초록색
        if vehicle_is_crash == 1:
            vehicle_color = (0, 165, 255)  # 주황색
        elif vehicle_is_crash == 2:
            vehicle_color = (0, 0, 255)  # 빨간색

        # 차량 바운딩 박스 그리기
        cv2.rectangle(frame, (vehicle_x1, vehicle_y1), (vehicle_x2, vehicle_y2), vehicle_color, 2)
        

        # 차량 주행 상태 판별 및 업데이트
        if vehicle_is_crash == 2:  # 충돌 직전인 경우에만 주행 상태를 판별하고 출력하고 저장
            current_bbox_size = (vehicle_x2 - vehicle_x1) * (vehicle_y2 - vehicle_y1)
            vehicle_is_driving = vehicle_detector.determine_driving_status(current_bbox_size)
            vehicle_detector.update_last_bbox_size(current_bbox_size)

            if vehicle_track_id:
                vehicle_pass_fail = vehicle_detector.determine_pass_fail_status(vehicle_is_driving)
                vehicle_detector.pass_fail_status_history[vehicle_track_id].append(vehicle_pass_fail)

                # 최종 10개 데이터 기반 판단
                if len(vehicle_detector.pass_fail_status_history[vehicle_track_id]) == 15:
                    vehicle_detector.pass_fail_status[vehicle_track_id] = mode(vehicle_detector.pass_fail_status_history[vehicle_track_id])
                else:
                    vehicle_detector.pass_fail_status[vehicle_track_id] = vehicle_pass_fail

                # 차량 주행 상태에 따라 표시할 메시지 설정
                vehicle_status_msg = "Driving" if vehicle_is_driving else "Stopped"
                vehicle_pass_fail = vehicle_detector.pass_fail_status[vehicle_track_id]

                # 화면에 차량 주행 상태 출력
                cv2.putText(frame, f"Vehicle ID: {vehicle_track_id} Status: {vehicle_status_msg} Pass/Fail: {vehicle_pass_fail}", 
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, vehicle_color, 2, cv2.LINE_AA)

                # 데이터프레임에 차량 주행 상태 저장
                if vehicle_class_id == 0:  # pedestrian인 경우
                    classification = "pedestrian"
                else:  # 나머지인 경우는 모두 vehicle로 처리
                    classification = "vehicle"

                df_data.append({'Classification': classification, 'Object_Type': vehicle_detector.classes.get(vehicle_class_id, None), 'ID': vehicle_track_id, 'Class ID' : vehicle_class_id,'Status': vehicle_status_msg, 'Pass/Fail': vehicle_pass_fail})
                # json 파일에도 추가
                time_formatted = format_time(cap.get(cv2.CAP_PROP_POS_MSEC))
                vehicle_result = {
                    "frame_number": frame_count,
                    "time": time_formatted,  # 현재 동영상 시간 (밀리초)
                    "object_type":  vehicle_detector.classes.get(vehicle_class_id, None),
                    "class_id": vehicle_class_id,
                    "status": vehicle_status_msg,
                    "pass_fail": vehicle_pass_fail,
                    "bounding_box": {
                        "x1": vehicle_x1,
                        "y1": vehicle_y1,
                        "x2": vehicle_x2,
                        "y2": vehicle_y2
                    }
                }
                vehicle_results.append(vehicle_result)

        # 신호등 감지
        detected_objects, results = traffic_light_detector.detect_traffic_light(frame)
        if results[0].boxes.is_track:
            for detection in results[0].boxes.data:
                traffic_x1, traffic_y1, traffic_x2, traffic_y2, obj_id, _, traffic_class_id = map(int, detection[:7])
                cv2.rectangle(frame, (traffic_x1, traffic_y1), (traffic_x2, traffic_y2), (0,255,0), 2)

        # 신호등 바운딩 박스 및 정보 출력
        for obj_id, size in detected_objects:
            # 신호등 바운딩 박스 및 정보 출력
            status = "DRIVING" if traffic_light_detector.check_driving_status(obj_id) else "STOP"
            signal_color = traffic_light_detector.signal_color[obj_id]
            
            # 데이터프레임에 신호등 정보 저장
            if signal_color is not None:
                pass_fail = traffic_light_detector.pass_fail_status[obj_id]
                cv2.putText(frame, f"Traffic Light ID: {obj_id} Status: {status}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Signal color: {signal_color}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Pass/Fail: {pass_fail}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                if pass_fail != "Calculating":
                # 데이터프레임에 신호등 정보 저장
                    df_data.append({'Classification':'traffic_light', 'Object_Type': traffic_light_detector.classes.get(traffic_class_id, None), 'ID': obj_id, 'Class ID' :traffic_class_id, 'Status': status, 'Signal Color': signal_color, 'Pass/Fail': pass_fail})
                    
                    # 신호등 결과 추가
                    time_formatted = format_time(cap.get(cv2.CAP_PROP_POS_MSEC))
                    traffic_light_result = {
                        "frame_number": frame_count,
                        "time": time_formatted,
                        "object_type": traffic_light_detector.classes.get(traffic_class_id, None),
                        "class_id": traffic_class_id,
                        "status": status,
                        "signal_color": signal_color,
                        "pass_fail": pass_fail,
                        "bounding_box": {
                            "x1": traffic_x1,
                            "y1": traffic_y1,
                            "x2": traffic_x2,
                            "y2": traffic_y2
                        }
                    }
                    traffic_light_results.append(traffic_light_result)

        # 화면에 프레임 출력
        out.write(frame)  # 결과 프레임을 파일에 저장
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()  # 비디오 파일 저장 완료
    cv2.destroyAllWindows()

    # 결과 데이터프레임 생성
    df = pd.DataFrame(df_data)

    # 중복된 ID에 대해 최다빈도수 값만 남기고 나머지는 제거
    # df = df.groupby('ID').agg(lambda x: Counter(x).most_common(1)[0][0]).reset_index()
    df = df.groupby('ID').apply(get_most_common_row).apply(pd.Series)

    # 데이터프레임을 CSV 파일로 저장
    df.to_csv(f"./data/output_data/output_{base_filename}.csv", index=False, header=True)
    
    # 데이터프레임을 JSON 파일로 저장
    save_results_to_json(vehicle_results, traffic_light_results, 'analysis_results.json', video_length)

if __name__ == "__main__":
    video_path = "./data/input_data/b.MOV"
    vehicle_model_path = "./src/model/lane_detect/yolov8n.pt"
    traffic_light_model_path = "./src/model/traffic_light/traffic_best_ver2.pt"
    threshold = 30  # 주행 상태 판단을 위한 임계값 설정
    main(video_path, vehicle_model_path, traffic_light_model_path)

