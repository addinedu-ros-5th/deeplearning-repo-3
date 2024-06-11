import sys
import os
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5.QtCore import *
import res_rc
import video
import tkinter as tk
import requests
import time


from_class = uic.loadUiType('login.ui')[0] 
sign_up_class = uic.loadUiType('signup.ui')[0] 

total_gui_class = uic.loadUiType('Window.ui')[0] 
video_class = uic.loadUiType('video.ui')[0] 

class FileUploadThread(QThread):
    update_progress = pyqtSignal(int)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, files, url):
        super().__init__()
        self.files = files
        self.url = url

    def run(self):
        response_files = []
        total_files = len(self.files)

        try:
            for i, file_path in enumerate(self.files):
                files = [('file', (os.path.basename(file_path), open(file_path, 'rb')))]
                response = requests.post(self.url, files=files)
                if response.status_code == 201:
                    response_files.extend(response.json().get('processed_videos', []))
                else:
                    self.error.emit(f"File transfer failed: {response.status_code}")
                    return

                progress = (i + 1) / total_files * 100
                self.update_progress.emit(int(progress))

        except Exception as e:
            self.error.emit(f"Error occurred while transferring file: {str(e)}")
            return

        self.finished.emit(response_files)
        
class VideoPlayer(QMainWindow, video_class):
    def __init__(self, video_path=None):
        super(VideoPlayer, self).__init__()
        self.setupUi(self)
        
        self.setWindowTitle("Visual Resualt") 
        
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.frame_position = 0
        self.playing = False
        self.memo_file_name = None 

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
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
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
            
            bytes_per_line = ch * new_width
            qt_image = QImage(resized_frame.data, new_width, new_height, bytes_per_line, QImage.Format_RGB888)
            self.view.setPixmap(QPixmap.fromImage(qt_image).scaled(self.view.size(), Qt.KeepAspectRatio))

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
                self.memo_file_name = file_name  

class Total_gui_Window(QMainWindow, total_gui_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        self.setWindowTitle("Main Window") 
        
        self.windows.clicked.connect(self.switch_to_windowPage)
        self.windows1.clicked.connect(self.switch_to_windowPage)
        self.sign1.clicked.connect(self.switch_login)
        self.sign.clicked.connect(self.switch_login) 
        self.icon_widget.setHidden(True)
        
        self.upload.clicked.connect(self.upload_files)
        self.analyze.clicked.connect(self.analyze_files)
        self.delbtn.clicked.connect(self.delete_selected_rows)
        self.search.clicked.connect(self.serach_from_table)
        self.searchbar.setPlaceholderText("Enter text to search")

        self.tableWidget.setColumnCount(4)
        self.tableWidget.setHorizontalHeaderLabels(['Check', 'File Path', 'File Name', 'Status'])
        self.tableWidget.cellClicked.connect(self.toggle_checkbox) 
        self.tableWidget.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        header = self.tableWidget.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tableWidget.setColumnHidden(1, True) 
        
        self.tableWidget.cellClicked.connect(self.cell_was_clicked)
        
        self.tableWidget1.setColumnCount(6)
        self.tableWidget1.setHorizontalHeaderLabels(['File path', 'Video ID', 'Speed', 'Pedestrian', 'Traffic', 'Fail Num'])
        self.tableWidget1.cellClicked.connect(self.cell_was_clicked)
        self.tableWidget1.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableWidget1.setColumnHidden(0, True)  
        
    def switch_to_windowPage(self):
        self.stackedWidget.setCurrentIndex(0)
    
    def upload_files(self):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self, "파일 선택", "", "비디오 파일 (*.mp4 *.avi *.mov *.mkv);;모든 파일 (*)", options=options)
        if files:
            for file in files:
                file_name = os.path.basename(file)  
                file_path = file
                for row in range(self.tableWidget.rowCount()):
                    existing_file_name = self.tableWidget.item(row, 2).text()
                    if file_name == existing_file_name:
                        reply = QMessageBox.question(self, 'Warning', f"'{file_name}' The file is already exiests.", 
                                                     QMessageBox.Yes)
                        if reply == QMessageBox.Yes:
                            self.upload_files() 
                        return
                else:  
                    row_position = self.tableWidget.rowCount()
                    self.tableWidget.insertRow(row_position)
                    self.tableWidget.setItem(row_position, 2, QTableWidgetItem(file_name))
                    self.tableWidget.setItem(row_position, 1, QTableWidgetItem(file_path)) 
                
                checkbox_widget = QWidget()
                checkbox = QCheckBox()
                checkbox.setEnabled(False)
                checkbox.setChecked(False)
                checkbox_layout = QHBoxLayout(checkbox_widget)
                checkbox_layout.addWidget(checkbox)
                checkbox_layout.setAlignment(Qt.AlignCenter)  
                checkbox_layout.setContentsMargins(0, 0, 0, 0)
                self.tableWidget.setCellWidget(row_position, 0, checkbox_widget)
    
    def delete_selected_rows(self):
        rows_to_delete = []
        for row in range(self.tableWidget.rowCount()):
            checkbox_widget = self.tableWidget.cellWidget(row, 0)
            checkbox = checkbox_widget.findChild(QCheckBox)
            if checkbox.isChecked():
                rows_to_delete.append(row)
        for row in reversed(rows_to_delete): 
            self.tableWidget.removeRow(row)
                
    def toggle_checkbox(self, row, column):
        if column == 2:  
            checkbox_widget = self.tableWidget.cellWidget(row, 0)
            checkbox = checkbox_widget.findChild(QCheckBox)
            checkbox.setChecked(not checkbox.isChecked())
    
    def serach_from_table(self):
        search_text = self.searchbar.text()
        if search_text == "":
            for row in range(self.tableWidget.rowCount()):
                self.tableWidget.setRowHidden(row, False)
        else:
            for row in range(self.tableWidget.rowCount()):
                file_name = self.tableWidget.item(row, 2).text()
                if search_text.lower() in file_name.lower():
                    self.tableWidget.setRowHidden(row, False)
                else:
                    self.tableWidget.setRowHidden(row, True)
            
    def analyze_files(self):
        row_count = self.tableWidget.rowCount()
        if row_count == 0:
            QMessageBox.warning(self, "Warning", "There are no files to analyze.")
            return

        files_to_analyze = []
        for row in range(self.tableWidget.rowCount()):
            file_path = self.tableWidget.item(row, 1).text()
            files_to_analyze.append(file_path)

        self.progress_dialog = QProgressDialog("Analyzing files...", "Cancel", 0, len(files_to_analyze), self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setValue(0)

        url = 'http://192.168.0.126:5000/api/upload'
        self.thread = FileUploadThread(files_to_analyze, url)
        self.thread.update_progress.connect(self.update_progress)
        self.thread.error.connect(self.on_error)
        self.thread.start()
        self.progress_dialog.canceled.connect(self.cancel_analysis)

    def update_progress(self, value):
        self.progress_dialog.setValue(value)
        if value == self.tableWidget.rowCount():
            self.progress_dialog.close()
            QMessageBox.information(self, "Successed", "All files have been analyzed.")

    def on_error(self, error_message):
        self.progress_dialog.close()
        QMessageBox.critical(self, "Error", error_message)

    def cancel_analysis(self):
        self.thread.terminate()
        self.progress_dialog.close() 

    def update_table_status(self, value, file_name, status):
        for row in range(self.tableWidget1.rowCount()):
            if self.tableWidget1.item(row, 1).text() == file_name:
                self.tableWidget1.setItem(row, 5, QTableWidgetItem(status))
                break
        self.update_progress(value)
    
    def cell_was_clicked(self, row, column):
        if column == 5:
            video_file_name = self.tableWidget1.item(row, 1).text()
            self.open_video_ui(video_file_name)
            
    def open_video_ui(self, video_file_name):
        self.other_window = VideoPlayer(video_file_name)
        self.other_window.show()

    def open_total_gui_window(self):
        self.total_gui_window = Total_gui_Window()
        self.total_gui_window.show()
        self.close()

    def switch_login(self):
        reply = QMessageBox.question(self, 'Message', 'Would you like to return to the login screen??',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.go_back()
        else:
            pass
        
    def go_back(self):
        self.from_class = WindowClass()
        self.from_class.show()
        self.close()
    
# -------------------------------------------------------------------------------------------------------------------------------------------

class Sign_up_Window(QDialog, sign_up_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.Sign.clicked.connect(self.sign_up)
        self.Check.clicked.connect(self.Check_id)
        
    def Check_id(self):
        user_id = self.Idld.text()
        
        data = {'user_id': user_id}
        response = requests.post('http://192.168.0.126:5000/api/check', json=data)
        
        if response.status_code == 401:
            
            QMessageBox.information(self, 'ID double check', 'This ID is already registered.', QMessageBox.Ok)
            return
        
        else:
            QMessageBox.information(self, 'ID Check', 'This ID is Available ID.', QMessageBox.Ok)
    
    def sign_up(self):
        
        user_birthday = self.Birthdayld.text()
        user_name = self.Nameld.text()
        user_id = self.Idld.text()
        user_password = self.Passwordld.text()
        check_password = self.Checkld.text()
        
        if user_password != check_password:
            QMessageBox.warning(self, "Password Check", "Passwords do not match.", QMessageBox.Ok)
            return 

        if len(user_birthday) > 7:
            QMessageBox.warning(self, 'Date of birth error', 'Please enter 6 characters for your date of birth.', QMessageBox.Ok)
            return
        
                
        if len(user_name) > 5:
            QMessageBox.warning(self, "Name error", "Please enter a name of 4 characters or less.", QMessageBox.Ok)
            return

        if len(user_id) > 10:
            QMessageBox.warning(self, "ID error", "Please enter ID with 10 characters or less.", QMessageBox.Ok)
            return

        if len(user_password) < 8 or len(user_password) > 20:
            QMessageBox.warning(self, "Password error", "Please enter your password between 8 and 20 characters.", QMessageBox.Ok)
            return
            
            
        data = {'user_birthday':user_birthday, 'user_name': user_name, 'user_id': user_id, 'user_password':user_password}
        response = requests.post('http://192.168.0.126:5000/api/signup', json=data)
        
        
        if response.status_code == 201:
            QMessageBox.information(self, 'registration successful', 'You have successfully registered.', QMessageBox.Ok)
            self.accept()

        elif response.status_code == 404:
            QMessageBox.information(self, 'registration fail', 'Unable to connect to server.', QMessageBox)
        
        else:
            QMessageBox.warning(self, 'registration fail', 'registration failed.', QMessageBox.Ok)
        print(response)
        
class WindowClass(QMainWindow, from_class) :
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        self.setWindowTitle("Driving Service")
        
        self.Signupbtn.clicked.connect(self.open_sign_window)
        self.Loginbtn.clicked.connect(self.Login_play)
        
    def Login_play(self):
        user_id = self.Idedit.text()
        user_password = self.Passwordedit.text()
        
        data = {'user_id': user_id, 'user_password':user_password}
        response = requests.post('http://192.168.0.126:5000/api/signin', json=data)
        
        
        if response.status_code == 201:
            QMessageBox.information(self, 'Login successful', 'You have successfully logged in.', QMessageBox.Ok)
            self.open_total_gui_window()
            
        elif response.status_code == 404:
            QMessageBox.information(self, 'Login failed', 'Unable to connect to server.', QMessageBox)
        
        else:
            QMessageBox.warning(self, 'Login failed', 'Login failed.', QMessageBox.Ok)
        print(response)
        
    def open_sign_window(self):
        self.new_window = Sign_up_Window()
        self.close()
        self.new_window.exec_()
        self.show()

    def open_total_gui_window(self):
        self.total_gui_window = Total_gui_Window()
        self.total_gui_window.show()
        self.close()
    
# class WorkerThread(QThread):
#     progress_changed = pyqtSignal(int, str, str)  # 진행률, 파일 이름, 상태

#     def __init__(self, files):
#         super().__init__()
#         self.files = files

#     def run(self):
#         for i, (file_path, file_name) in enumerate(self.files):
#             try:
#                 with open(file_path, 'rb') as f:
#                     server_address = 'http://192.168.1.214:5000'
#                     response = requests.post(f'{server_address}/api/upload', files={'file': f})
#                 if response.status_code == 200:
#                     self.progress_changed.emit(i + 1, file_name, "analyzed")
#                 else:
#                     self.progress_changed.emit(i + 1, file_name, "failed")
#             except Exception as e:
#                 self.progress_changed.emit(i + 1, file_name, "failed")
#                 print(f"Error analyzing {file_name}: {e}")

       
if __name__ == "__main__":
    app = QApplication(sys.argv) 
    myWindows = WindowClass() 
    # myWindows = Total_gui_Window() 
    myWindows.show() 
    
    sys.exit(app.exec_()) 
