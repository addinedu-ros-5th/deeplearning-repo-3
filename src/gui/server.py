import mysql.connector
from flask_cors import CORS
import base64
from flask import *
from werkzeug.utils import secure_filename
import os
import cv2
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)
CORS(app)

# MySQL 데이터베이스 연결 설정
db = mysql.connector.connect(
    host="database-1.czkmo68qelp7.ap-northeast-2.rds.amazonaws.com",
    user="mk",
    password="1234",
    database="Driving"
)

UPLOAD_FOLDER = '/home/hb/dev_ws/running/deep/project'
PROCESSED_FOLDER = '/home/hb/Downloads/project_test'  # 실제 처리된 파일을 저장할 폴더 경로로 변경하십시오
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Load the YOLOv8 model
model = YOLO('/home/hb/dev_ws/running/deep/project/Logic/all_best.pt')

# 신호등 클래스 ID를 설정합니다. (예시로 6번 클래스 ID로 설정)
traffic_light_class_id = 6

# 신호등 색상 예측을 위한 모델을 로드합니다.
traffic_light_color_model = YOLO("/home/hb/dev_ws/running/deep/project/Logic/traffic_best.pt")

# 신뢰도 임계값 설정
confidence_threshold = 0.2

def detect_traffic_light(frame):
    results = model(frame)
    for result in results:
        for detection in result.boxes.data:
            x1, y1, x2, y2, confidence, class_id = detection
            confidence = float(confidence)
            class_id = int(class_id)
            print(f"Detected class ID: {class_id}, Confidence: {confidence}")
            if class_id == traffic_light_class_id and confidence >= confidence_threshold:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                traffic_light_roi = frame[y1:y2, x1:x2]
                is_green, is_red = is_green_or_red_light(traffic_light_roi)
                color = (0, 255, 0) if is_green else (0, 0, 255) if is_red else (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = "Green Light" if is_green else "Red Light" if is_red else "Unknown"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                if is_green:
                    return True
    return False

def is_green_or_red_light(traffic_light_roi):
    hsv = cv2.cvtColor(traffic_light_roi, cv2.COLOR_BGR2HSV)
    green_lower = np.array([33, 30, 30])
    green_upper = np.array([89, 255, 255])
    red_lower1 = np.array([0, 70, 70])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 70, 70])
    red_upper2 = np.array([180, 255, 255])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_upper2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    green_ratio = cv2.countNonZero(green_mask) / (traffic_light_roi.size / 3)
    red_ratio = cv2.countNonZero(red_mask) / (traffic_light_roi.size / 3)
    print(f"Green ratio: {green_ratio}, Red ratio: {red_ratio}")
    is_green = green_ratio > 0.05
    is_red = red_ratio > 0.05
    return is_green, is_red





@app.route('/api/send_data', methods=['POST'])
def insert_pass_to_database_local(video_name):
    try:
        cursor = db.cursor()
        sql = "INSERT INTO violation (Video_ID, Speed, Pedestrian, Traffic, Fail_Num) VALUES (%s, 'pass', 'pass', 'pass', 'pass')"
        cursor.execute(sql, (video_name,))
        db.commit()
        print("Pass 문장이 데이터베이스에 삽입되었습니다.")
        
        data_to_send = {
            "Video_ID": video_name,
            "Speed": "pass",
            "Pedestrian": "pass",
            "Traffic": "pass",
            "Fail_Num": "pass"
        }
        return jsonify(data_to_send), 200
        
    except Exception as e:
        print("Pass 문장 삽입 중 오류가 발생했습니다:", e)
        return None


def evaluate_traffic_lights(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    video_name = os.path.basename(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        is_green_light_detected = detect_traffic_light(frame)
        if is_green_light_detected:
            print("정차 상태에서 초록불에 출발했습니다.")
        else:
            print("정차 상태에서 초록불에 출발하지 않았습니다.")
        out.write(frame)
    cap.release()
    out.release()
    insert_pass_to_database_local(video_name)  # 함수 호출 시 video_name 전달
   
   
@app.route('/api/process_video/<filename>', methods=['GET'])
def process_and_send_video(filename):
    # 비디오 파일 경로 생성
    processed_video_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    
    # 파일이 존재하는지 확인
    if not os.path.exists(processed_video_path):
        return "File not found", 404
    
    # 처리된 비디오 파일을 클라이언트에게 전송
    return send_file(processed_video_path)

def process_video(video_path):
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], 'annotated_' + os.path.basename(video_path))
    evaluate_traffic_lights(video_path, output_path)  # 비디오 처리 함수 호출
    return 'annotated_' + os.path.basename(video_path)
    
    
    
# 초기값은 처리 중으로 설정
completion_status = 'processing'

# 각 영상 파일의 처리 상태를 저장하는 딕셔너리
video_completion_statuses = {}

@app.route('/api/completion_status', methods=['GET'])
def get_completion_status():
    global completion_status
    # 모든 영상 파일의 처리 상태를 확인하여 모두가 'completed' 상태이면 'completed'로 설정
    if all(status == 'completed' for status in video_completion_statuses.values()):
        completion_status = 'completed'
    return jsonify({'completion_status': completion_status})

@app.route('/api/set_completion_status', methods=['POST'])
def set_completion_status():
    global completion_status
    data = request.get_json()
    new_status = data.get('status')
    completion_status = new_status
    return jsonify({'message': 'Completion status updated successfully', 'completion_status': completion_status}), 200

processed_results = []
new_results_index = 0



@app.route('/api/upload', methods=['POST'])
def upload_video():

    global processed_results 
	
    if 'file' not in request.files:
        return jsonify({'error': 'No video files provided'}), 400

    videos = request.files.getlist('file')
    if not videos:
        return jsonify({'error': 'No selected files'}), 405

    processed_videos = []

    for video in videos:
        if video.filename == '':
            continue

        filename = secure_filename(video.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video.save(filepath)

        processed_video_path = process_video(filepath)
        processed_videos.append(processed_video_path)
        
        
        
        processed_results.append({
            "Video_ID": filename,
            "Speed": "pass",
            "Pedestrian": "pass",
            "Traffic": "pass",
            "Fail_Num": "0"
        })
    
    

    processed_video_urls = [url_for('process_and_send_video', filename=os.path.basename(path), _external=True) for path in processed_videos]
    
    
    return jsonify({'message': 'Files uploaded and processed successfully', 'processed_videos': processed_video_urls}), 201


@app.route('/api/processing_complete', methods=['GET'])
def processing_complete():
    global processed_results, new_results_index  # 전역 변수를 사용할 것을 명시
    if new_results_index < len(processed_results):
        # 새로운 결과가 있는 경우 해당 결과를 반환하고 인덱스 업데이트
        new_data = processed_results[new_results_index:]
        new_results_index = len(processed_results)
        return jsonify(new_data), 200
    else:
        return jsonify([]), 200 


#----------------------------------------------------------------------------------------------------------------------------------

@app.route('/api/check', methods=['POST'])
def check_id():
    data = request.get_json()
    id = data.get('user_id')
    cursor = db.cursor()
    cursor.execute("SELECT * FROM member WHERE ID = %s", (id,))
    user = cursor.fetchone()

    if user:
        return jsonify({'eerror': '중복된 아이디 임다'}), 401
    else:
        return jsonify({'messag': '사용 가능'}), 201

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json()
    id = data.get('user_id')
    password = data.get('user_password')
    hash_password = base64.b64encode(password.encode('utf-8')).decode('utf-8')
    name = data.get('user_name')
    birthday = data.get('user_birthday')
    # 데이터베이스에서 사용자가 이미 존재하는지 확인
    cursor = db.cursor()
    cursor.execute("SELECT * FROM member WHERE ID = %s", (id,))
    user = cursor.fetchone()

    if user:
        return jsonify({'error': 'Username already exists'}), 400
    
    # 새로운 사용자 생성
    cursor.execute("INSERT INTO member (Birthday, Name, ID, Password) VALUES (%s, %s, %s, %s)", (birthday, name, id, hash_password))
    db.commit()
    return jsonify({'message': '회원가입 완료'}), 201

@app.route('/api/signin', methods=['POST'])
def signin():
    data = request.get_json()

    id = data.get('user_id')
    password = data.get('user_password')
    hash_password = base64.b64encode(password.encode('utf-8')).decode('utf-8')
    cursor = db.cursor()
    cursor.execute("SELECT * FROM member WHERE ID = %s AND Password = %s", (id, hash_password))
    user = cursor.fetchone()
    if user:
        return jsonify({'message': '로그인 완료'}), 201
    else:
        return jsonify({'error': 'Invalid ID or password'}), 401

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
