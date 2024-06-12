import mysql.connector
from flask_cors import CORS
import base64
from flask import Flask, request, jsonify, Response, send_file
from werkzeug.utils import secure_filename
import os
import cv2
from ultralytics import YOLO
from judge_logic.total_evaluation.total_evaluation import main as evaluation_main
import zipfile

app = Flask(__name__)
CORS(app)

# MySQL 데이터베이스 연결 설정
db = mysql.connector.connect(
    host="database-1.czkmo68qelp7.ap-northeast-2.rds.amazonaws.com",
    user="mk",
    password="1234",
    database="Driving"
)

UPLOAD_FOLDER = '/home/addinedu/dev_ws/src/video'
PROCESSED_FOLDER = '/data/output_data'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load the YOLOv8 model
model = YOLO('/home/addinedu/dev_ws/src/YOLO/yolov8n.pt')

@app.route('/api/upload', methods=['POST'])
def upload_video():
    if 'videos' not in request.files:
        return jsonify({'error': 'No video files provided'}), 400

    videos = request.files.getlist('videos')
    if not videos:
        return jsonify({'error': 'No selected files'}), 400

    processed_videos = []

    for video in videos:
        if video.filename == '':
            continue

        filename = secure_filename(video.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        video.save(filepath)

        # Process the video and create the necessary files
        processed_video_path, json_path, csv_path = process_video(filepath)

        # Create a zip file containing the processed files
        zip_path = create_zip_file(processed_video_path, json_path, csv_path)
        processed_videos.append(zip_path)

    return jsonify({'message': 'Files uploaded and processed successfully', 'processed_videos': processed_videos}), 201

def process_video(video_path):
    base_filename = os.path.splitext(os.path.basename(video_path))[0]
    output_video_path = os.path.join(PROCESSED_FOLDER, f'output_{base_filename}.mp4')
    json_path = os.path.join(PROCESSED_FOLDER, f'{base_filename}.json')
    csv_path = os.path.join(PROCESSED_FOLDER, f'{base_filename}.csv')

    # Call the main function from total_evaluation.py
    evaluation_main(video_path, model.model_path, model.model_path)

    return output_video_path, json_path, csv_path

def create_zip_file(video_path, json_path, csv_path):
    base_filename = os.path.splitext(os.path.basename(video_path))[0]
    zip_filename = os.path.join(PROCESSED_FOLDER, f'{base_filename}.zip')
    
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        zipf.write(video_path, os.path.basename(video_path))
        zipf.write(json_path, os.path.basename(json_path))
        zipf.write(csv_path, os.path.basename(csv_path))
    
    return zip_filename

@app.route('/api/download/<zip_name>', methods=['GET'])
def download_zip(zip_name):
    zip_path = os.path.join(PROCESSED_FOLDER, f'{zip_name}.zip')
    
    if not os.path.exists(zip_path):
        return jsonify({'error': 'File not found'}), 404

    return send_file(zip_path, as_attachment=True), 200


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
