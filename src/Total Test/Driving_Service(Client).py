import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5.QtCore import *
import requests
import res_rc
import video
import time
import tkinter as tk


from_class = uic.loadUiType('/home/ys/Downloads/zDeepgui_rev1/login.ui')[0] # 현복 login ui파일 
sign_up_class = uic.loadUiType('/home/ys/Downloads/zDeepgui_rev1/signup.ui')[0] # 현복 Sign_up ui 파일

total_gui_class = uic.loadUiType('/home/ys/Downloads/zDeepgui_rev1/Window.ui')[0] # 영수님 Main_ui파일

# 실시간 데이터 전송 업데이트 Class
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
                    self.error.emit(f"파일 전송 실패: {response.status_code}")
                    return

                progress = (i + 1) / total_files * 100
                self.update_progress.emit(int(progress))  # 신호를 발생시킵니다.

        except Exception as e:
            self.error.emit(f"파일 전송 중 오류 발생: {str(e)}")
            return

        self.finished.emit(response_files)



# Total GUI 화면 클래스 (영수님 mainwindow file 복붙 ㄱㄱ)
class Total_gui_Window(QMainWindow, total_gui_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        self.setWindowTitle("Main Window") 
        
        # 메뉴용 버튼
        self.windows.clicked.connect(self.switch_to_windowPage)
        self.windows1.clicked.connect(self.switch_to_windowPage)
        # self.details.clicked.connect(self.switch_to_detailsPage)
        # self.details1.clicked.connect(self.switch_to_detailsPage)
        # self.set.clicked.connect(self.switch_to_settingsPage)
        # self.set1.clicked.connect(self.switch_to_settingsPage)
        self.sign1.clicked.connect(self.switch_login) # Logout 버튼
        self.sign.clicked.connect(self.switch_login) # Logout 버튼
        self.icon_widget.setHidden(True)
        
        # 윈도우창 내 버튼용
        self.upload.clicked.connect(self.upload_files)
        self.analyze.clicked.connect(self.analyze_files)
        self.delbtn.clicked.connect(self.delete_selected_rows)
        self.search.clicked.connect(self.serach_from_table)
        self.searchbar.setPlaceholderText("Enter text to search")

        self.tableWidget.setColumnCount(4)
        self.tableWidget.setHorizontalHeaderLabels(['선택', '파일경로', '파일 이름', '상태'])
        self.tableWidget.cellClicked.connect(self.toggle_checkbox)
        self.tableWidget.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        header = self.tableWidget.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tableWidget.setColumnHidden(1, True)  # 파일 경로 열 숨기기
        
        self.tableWidget.cellClicked.connect(self.cell_was_clicked)
        
        self.tableWidget1.setColumnCount(6)
        self.tableWidget1.setHorizontalHeaderLabels(['파일경로', 'Video ID', 'Speed', 'Pedestrian', 'Traffic', 'Fail Num'])
        self.tableWidget1.cellClicked.connect(self.cell_was_clicked)
        self.tableWidget1.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableWidget1.setColumnHidden(0, True)  # 파일 경로 열 숨기기
        
    #메뉴 배너용 창 전환 설정
    def switch_to_windowPage(self):
        self.stackedWidget.setCurrentIndex(0)
        
    # def switch_to_detailsPage(self):
    #     self.stackedWidget.setCurrentIndex(1)
        
    # def switch_to_settingsPage(self):
    #     self.stackedWidget.setCurrentIndex(2)
    
    #윈도우창의 uplaod 버튼용
    def upload_files(self):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self, "파일 선택", "", "비디오 파일 (*.mp4 *.avi *.mov *.mkv);;모든 파일 (*)", options=options)
        if files:
            for file in files:
                file_name = os.path.basename(file)  # 파일 이름 추출
                file_path = file
                # 이미 존재하는 파일인지 확인
                for row in range(self.tableWidget.rowCount()):
                    existing_file_name = self.tableWidget.item(row, 2).text()
                    if file_name == existing_file_name:
                        reply = QMessageBox.question(self, '경고', f"'{file_name}' 파일은 이미 존재합니다.", 
                                                     QMessageBox.Yes)
                        if reply == QMessageBox.Yes:
                            self.upload_files()  # 파일 탐색기 대화 상자 다시 열기
                        return
                else:  # 이미 존재하지 않는 파일인 경우에만 테이블 위젯에 추가
                    row_position = self.tableWidget.rowCount()
                    self.tableWidget.insertRow(row_position)
                    self.tableWidget.setItem(row_position, 2, QTableWidgetItem(file_name))
                    self.tableWidget.setItem(row_position, 1, QTableWidgetItem(file_path))  # 파일 경로 추가
                
                # 체크 박스 설정
                checkbox_widget = QWidget()
                checkbox = QCheckBox()
                checkbox.setEnabled(False)  # 사용자가 선택할 수 없도록 설정
                checkbox.setChecked(False)
                checkbox_layout = QHBoxLayout(checkbox_widget)
                checkbox_layout.addWidget(checkbox)
                checkbox_layout.setAlignment(Qt.AlignCenter)  # 가운데 정렬 설정
                checkbox_layout.setContentsMargins(0, 0, 0, 0)
                self.tableWidget.setCellWidget(row_position, 0, checkbox_widget)
    
    def delete_selected_rows(self):
        rows_to_delete = []
        for row in range(self.tableWidget.rowCount()):
            checkbox_widget = self.tableWidget.cellWidget(row, 0)
            checkbox = checkbox_widget.findChild(QCheckBox)
            if checkbox.isChecked():
                rows_to_delete.append(row)
        for row in reversed(rows_to_delete):  # 역순으로 삭제하여 인덱스 문제 방지
            self.tableWidget.removeRow(row)
                
    def toggle_checkbox(self, row, column):
        if column == 2:  # 파일 경로 열 클릭 시 체크박스 토글
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
            QMessageBox.warning(self, "경고", "분석할 파일이 없습니다.")
            return

        files_to_analyze = []
        for row in range(self.tableWidget.rowCount()):
            file_path = self.tableWidget.item(row, 1).text()
            # file_name = self.tableWidget.item(row, 2).text()
            files_to_analyze.append(file_path)

        # 진행 바 창 설정
        self.progress_dialog = QProgressDialog("Analyzing files...", "Cancel", 0, len(files_to_analyze), self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setValue(0)

        url = 'http://192.168.0.126:5000/api/upload'
        self.thread = FileUploadThread(files_to_analyze, url)
        self.thread.update_progress.connect(self.update_progress)
        #self.thread.finished.connect(self.on_finished)
        self.thread.error.connect(self.on_error)
        self.thread.start()
        self.progress_dialog.canceled.connect(self.cancel_analysis)

    def update_progress(self, value):
        self.progress_dialog.setValue(value)
        if value == self.tableWidget.rowCount():
            self.progress_dialog.close()
            QMessageBox.information(self, "완료", "모든 파일 분석이 완료되었습니다.")
            
    # def on_finished(self, response_files):
    #     self.progress_dialog.close()
    #     QMessageBox.information(self, "Success", "파일 전송이 완료되었습니다.")
    #     # response_files를 처리하는 코드를 여기에 작성하십시오.

    def on_error(self, error_message):
        self.progress_dialog.close()
        QMessageBox.critical(self, "Error", error_message)

    def cancel_analysis(self):
        self.thread.terminate()
        self.progress_dialog.close()  # 진행 바 창 닫기

    def update_table_status(self, value, file_name, status):
        for row in range(self.tableWidget.rowCount()):
            if self.tableWidget.item(row, 2).text() == file_name:
                self.tableWidget.setItem(row, 3, QTableWidgetItem(status))
                break
        self.update_progress(value)
    
    #분석 테이블에서 특정 셀 클릭 시 비디오창이 열리게 하는 코드    
    def cell_was_clicked(self, row, column):
        # 3번째 열 (index 2) 클릭 시 다른 UI 파일 열기
        if column == 3:
            self.open_video_ui()
            
    # def open_video_ui(self):
    #     self.other_window = video.VideoPlayer()
    #     self.other_window.show()

        # Total gui 화면 창 전환 (로그인 완료)
    def open_total_gui_window(self):
        self.total_gui_window = Total_gui_Window()
        self.total_gui_window.show()
        self.close()
    
    # Logout 기능
    def switch_login(self):
        reply = QMessageBox.question(self, 'Message', '로그인 화면으로 돌아가시겠습니까?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.go_back()
        else:
            pass
    

#--------------------------------------------------------------------------------------------------------------------------------
    # def analyze_files(self):
    #     if not self.files:
    #         QMessageBox.warning(self, "Warning", "먼저 파일을 업로드하십시오.")
    #         return

    #     url = 'http://192.168.0.156:5000/api/upload'
    #     self.progress_dialog = QProgressDialog("파일 전송 중...", None, 0, 100, self)
    #     self.progress_dialog.setWindowTitle("전송 중")
    #     self.progress_dialog.setWindowModality(Qt.WindowModal)
    #     self.progress_dialog.show()

    #     self.thread = FileUploadThread(self.files, url)
    #     self.thread.update_progress.connect(self.update_progress)
    #     self.thread.finished.connect(self.on_finished)
    #     self.thread.error.connect(self.on_error)
    #     self.thread.start()

    # def update_progress(self, progress):
    #     self.progress_dialog.setValue(progress)

    # def on_finished(self, response_files):
    #     self.progress_dialog.close()
    #     QMessageBox.information(self, "Success", "파일 전송이 완료되었습니다.")
    #     # response_files를 처리하는 코드를 여기에 작성하십시오.

    # def on_error(self, error_message):
    #     self.progress_dialog.close()
    #     QMessageBox.critical(self, "Error", error_message)



    # def receive_processed_files(self, file_names):
    #     base_url = 'http://192.168.0.156:5000/download'  # 처리된 파일을 받는 서버의 엔드포인트
    #     processed_files = []

    #     progress_dialog = QProgressDialog("파일 다운로드 중...", None, 0, len(file_names), self)
    #     progress_dialog.setWindowTitle("다운로드 중")
    #     progress_dialog.setWindowModality(Qt.WindowModal)

    #     for file_name in file_names:
    #         try:
    #             response = requests.get(f"{base_url}/{file_name}")
    #             if response.status_code == 200:
    #                 save_path = os.path.join("/home/hb/Downloads/project_test/", file_name)  # 처리된 파일을 저장할 경로 설정
    #                 with open(save_path, 'wb') as f:
    #                     f.write(response.content)
    #                 processed_files.append(save_path)
    #             else:
    #                 QMessageBox.warning(self, "Failed", f"{file_name} 파일 다운로드 실패: {response.status_code}")
    #         except Exception as e:
    #             QMessageBox.critical(self, "Error", f"파일 다운로드 중 오류 발생: {str(e)}")

# -------------------------------------------------------------------------------------------------------------------------------------------


# Sign up Class
class Sign_up_Window(QDialog, sign_up_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.Sign.clicked.connect(self.sign_up)
        self.Check.clicked.connect(self.Check_id)
        
    # ID 중복 체크
    def Check_id(self):
        user_id = self.Idld.text()
        
        data = {'user_id': user_id}
        response = requests.post('http://192.168.0.156:5000/api/check', json=data)
        
        if response.status_code == 401:
            
            QMessageBox.information(self, 'ID 중복 확인', '해당 ID는 이미 등록되어있습니다.', QMessageBox.Ok)
            return
        
        else:
            QMessageBox.information(self, 'ID 확인', '사용 가능한 ID입니다.', QMessageBox.Ok)
    
    # 회원 가입
    def sign_up(self):
        
        user_birthday = self.Birthdayld.text()
        user_name = self.Nameld.text()
        user_id = self.Idld.text()
        user_password = self.Passwordld.text()
        check_password = self.Checkld.text()
        
        if user_password != check_password:
            QMessageBox.warning(self, "비밀번호 확인", "비밀번호가 일치하지 않습니다.", QMessageBox.Ok)
            return 

        if len(user_birthday) > 7:
            QMessageBox.warning(self, '생년월일 오류', '생년월일은 6자로 입력해주세요.', QMessageBox.Ok)
            return
        
                
        if len(user_name) > 5:
            QMessageBox.warning(self, "이름 오류", "이름은 4자 이하로 입력해주세요.", QMessageBox.Ok)
            return

        if len(user_id) > 10:
            QMessageBox.warning(self, "ID 오류", "ID는 10자 이하로 입력해주세요.", QMessageBox.Ok)
            return

        if len(user_password) < 8 or len(user_password) > 20:
            QMessageBox.warning(self, "비밀번호 오류", "비밀번호는 8자 이상 20자 이하로 입력해주세요.", QMessageBox.Ok)
            return
            
            
        data = {'user_birthday':user_birthday, 'user_name': user_name, 'user_id': user_id, 'user_password':user_password}
        response = requests.post('http://192.168.0.156:5000/api/signup', json=data)
        
        
        if response.status_code == 201:
            QMessageBox.information(self, '회원가입 성공', '회원가입에 성공하셨습니다.', QMessageBox.Ok)
            self.accept()

        elif response.status_code == 404:
            QMessageBox.information(self, '회원가입 실패', '서버에 연결 할 수 없습니다.', QMessageBox)
        
        else:
            QMessageBox.warning(self, '회원가입 실패', '회원가입에 실패하였습니다.', QMessageBox.Ok)
        print(response)
        
        
# Main 화면 Class   
class WindowClass(QMainWindow, from_class) :
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        self.setWindowTitle("Driving Service")
        
        self.Signupbtn.clicked.connect(self.open_sign_window)
        self.Loginbtn.clicked.connect(self.Login_play)
        
    # 로그인
    def Login_play(self):
        user_id = self.Idedit.text()
        user_password = self.Passwordedit.text()
        
        data = {'user_id': user_id, 'user_password':user_password}
        response = requests.post('http://192.168.0.156:5000/api/signin', json=data)
        
        
        if response.status_code == 201:
            QMessageBox.information(self, '로그인 성공', '로그인에 성공하셨습니다.', QMessageBox.Ok)
            self.open_total_gui_window()
            
        elif response.status_code == 404:
            QMessageBox.information(self, '로그인 실패', '서버에 연결 할 수 없습니다.', QMessageBox)
        
        else:
            QMessageBox.warning(self, '로그인 실패', '로그인에 실패하였습니다.', QMessageBox.Ok)
        print(response)
        
    # 회원가입 창 전환
    def open_sign_window(self):
        self.new_window = Sign_up_Window()
        self.close()
        self.new_window.exec_()
        self.show()

    # Total gui 화면 창 전환 (로그인 완료)
    def open_total_gui_window(self):
        self.total_gui_window = Total_gui_Window()
        self.total_gui_window.show()
        self.close()
    
       
if __name__ == "__main__":
    app = QApplication(sys.argv) 
    # myWindows = WindowClass() 
    myWindows = WindowClass() 
    myWindows.show() 
    
    sys.exit(app.exec_()) 
