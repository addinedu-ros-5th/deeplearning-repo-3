from PyQt5 import QtWidgets, uic
import sys
import os
import cv2
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap

class VideoPlayer(QtWidgets.QMainWindow):
    def __init__(self):
        super(VideoPlayer, self).__init__()
        uic.loadUi('/home/ys/Downloads/zDeepgui_rev1/video.ui', self)
        self.setWindowTitle("Visual Resualt") 
        
        self.video_path = '40km-h_.mp4'
        self.cap = cv2.VideoCapture(self.video_path)
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.frame_position = 0
        self.playing = False
        self.memo_file_name = None  # 메모 파일 경로

        self.btn_play.clicked.connect(self.play_pause)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_forward.clicked.connect(self.forward)
        self.btn_prev.clicked.connect(self.prev)
        self.bar.sliderMoved.connect(self.set_position)
        self.Save.clicked.connect(self.save_memo)

        self.load_video(self.video_path)

    def load_video(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.bar.setMaximum(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    def play_pause(self):
        if self.playing:
            self.timer.stop()
        else:
            self.timer.start(30)
        self.playing = not self.playing

    def stop(self):
        self.timer.stop()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_position = 0
        self.bar.setValue(0)
        self.playing = False

    def forward(self):
        self.frame_position += 10 * int(self.cap.get(cv2.CAP_PROP_FPS))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_position)

    def prev(self):
        self.frame_position -= 10 * int(self.cap.get(cv2.CAP_PROP_FPS))
        if self.frame_position < 0:
            self.frame_position = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_position)

    def set_position(self, position):
        self.frame_position = position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_position)

    def next_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame_position += 1
            self.bar.setValue(self.frame_position)
            
            # Convert the frame to RGB format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize the frame while keeping the aspect ratio
            h, w, ch = frame.shape
            label_height, label_width = self.view.height(), self.view.width()
            aspect_ratio = w / h
            if label_width / aspect_ratio <= label_height:
                new_width = label_width
                new_height = int(label_width / aspect_ratio)
            else:
                new_height = label_height
                new_width = int(label_height * aspect_ratio)
            
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Create a QImage and set it to the QLabel
            bytes_per_line = ch * new_width
            qt_image = QImage(resized_frame.data, new_width, new_height, bytes_per_line, QImage.Format_RGB888)
            self.view.setPixmap(QPixmap.fromImage(qt_image).scaled(self.view.size(), Qt.KeepAspectRatio))

            # Update time label
            total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            current_time = QTime(0, 0).addSecs(int(self.frame_position / fps))
            total_time = QTime(0, 0).addSecs(int(total_frames / fps))
            self.time.setText(f'{current_time.toString("mm:ss")} / {total_time.toString("mm:ss")}')
        else:
            self.stop()

    def save_memo(self):
        if self.memo_file_name:
            try:
                with open(self.memo_file_name, 'w') as file:
                    memo_text = self.memotext.toPlainText()
                    file.write(memo_text)
                QMessageBox.information(self, "Success", "Memo saved successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred while saving the memo: {str(e)}")
        else:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Memo", "", "Text Files (*.txt);;All Files (*)", options=options)
            if file_name:
                memo_text = self.memotext.toPlainText()
                with open(file_name, 'w') as file:
                    file.write(memo_text)
                self.memo_file_name = file_name  # 새로 저장된 파일 경로 저장
    
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())
