import mysql.connector
from flask_cors import CORS
import base64
from flask import Flask, request, jsonify, Response, send_file
from werkzeug.utils import secure_filename
import os
import cv2
from ultralytics import YOLO
app = Flask(__name__)
CORS(app)
# MySQL 데이터베이스 연결 설정
db = mysql.connector.connect(
    host="database-1.czkmo68qelp7.ap-northeast-2.rds.amazonaws.com",
    user="mk",
    password="1234",
    database="Driving"
)
##--------------------
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the YOLOv8 model
model = YOLO('/home/addinedu/dev_ws/src/YOLO/yolov8n.pt')

@app.route('/api/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Process the video and save the annotated video
    process_video(filepath)

    return jsonify({'message': 'File uploaded and processed successfully'}), 201

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Get the frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter('annotated_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Perform object detection and annotation
        results = model.track(frame, persist=True)
        annotated_frame = results[0].plot()

        # Write the annotated frame to the output videoannotated_video.mp4
        out.write(annotated_frame)

    # Release everything when job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

@app.route('/api/download', methods=['GET'])
def download_video():
    video_path = '/home/addinedu/dev_ws/src/annotated_video.mp4'
    return send_file(video_path, as_attachment=True), 201

##-------------------------------
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
