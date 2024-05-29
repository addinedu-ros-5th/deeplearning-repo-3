import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5.QtCore import *
# import res2_rc
import requests
import time

from_class = uic.loadUiType('/home/ys/Downloads/login.ui')[0] # 현복 login ui파일 
sign_up_class = uic.loadUiType('/home/ys/Downloads//signup.ui')[0] # 현복 Sign_up ui 파일

total_gui_class = uic.loadUiType('/home/ys/Downloads/Window.ui')[0] # 영수님 Main_ui파일

# Total GUI 화면 클래스 (영수님 mainwindow file 복붙 ㄱㄱ)
class Total_gui_Window(QMainWindow, total_gui_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        self.setWindowTitle("Main Window") 
        
        self.windows.clicked.connect(self.switch_to_windowPage)
        self.windows1.clicked.connect(self.switch_to_windowPage)
        self.file.clicked.connect(self.switch_to_filesPage)
        self.file1.clicked.connect(self.switch_to_filesPage)
        self.details.clicked.connect(self.switch_to_detailsPage)
        self.details1.clicked.connect(self.switch_to_detailsPage)
        self.set.clicked.connect(self.switch_to_settingsPage)
        self.set1.clicked.connect(self.switch_to_settingsPage)
        
        self.upload.clicked.connect(self.upload_files)
        self.analyze.clicked.connect(self.analyze_files)
        
        self.model1 = QStringListModel()
        self.filelist.setModel(self.model1)
        self.model2 = QStringListModel()
        self.filelist1.setModel(self.model2)
        
        self.model = QFileSystemModel()
        self.model.setRootPath('home')
        self.fileTreeView.setModel(self.model)
        self.fileTreeView.setRootIndex(self.model.index('home'))
        self.fileTreeView.setColumnWidth(0, 250)
        
        self.files = []
    
    def switch_to_windowPage(self):
        self.stackedWidget.setCurrentIndex(0)
        
    def switch_to_filesPage(self):
        self.stackedWidget.setCurrentIndex(1)
        
    def switch_to_detailsPage(self):
        self.stackedWidget.setCurrentIndex(2)
        
    def switch_to_settingsPage(self):
        self.stackedWidget.setCurrentIndex(4)
    
    def upload_files(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "동영상 파일 선택", "", "동영상 파일 (*.mp4 *.avi *.mov *.mkv);;모든 파일 (*)", options=options)
        if files:
            self.files = files
            self.model1.setStringList(files) 


    def analyze_files(self):
        if not self.files:
            QMessageBox.warning(self, "Warning", "먼저 파일을 업로드하십시오.")
            return

        url = 'http://192.168.0.156:5000/api/video'  # 대상 PC의 IP 주소와 포트를 설정하십시오.
        response_files = []
        
        progress_dialog = QProgressDialog("파일 전송 중...", None, 0, len(self.files), self)
        progress_dialog.setWindowTitle("전송 중")
        progress_dialog.setWindowModality(Qt.WindowModal)
        
        for file_path in self.files:
            try:
                with open(file_path, 'rb') as f:
                    files = {'file': (os.path.basename(file_path), f)}
                    response = requests.post(url, files=files)
                    if response.status_code == 200:
                        response_files.append(os.path.basename(file_path))
                    else:
                        QMessageBox.warning(self, "Failed", f"{os.path.basename(file_path)} 파일 전송 실패: {response.status_code}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"파일 전송 중 오류 발생: {str(e)}")
        
        # # 서버로부터 파일이 처리된 후 다시 받기
        # self.receive_processed_files(response_files) 파일 전체 업로드 후 순차적으로 분석해서 가져오기 그 후 마지막에 그래프로 나타내기

    # def receive_processed_files(self, file_names):
    #     # 서버에서 파일이 처리될 때까지 기다리는 동안 대기 (여기서는 단순히 10초 대기)
    #     time.sleep(10)  # 실제로는 서버에서 완료를 알려주는 이벤트를 기다려야 함

    #     url = 'http://192.168.0.126:9200//home/ys/Downloads/recd/'  # 처리된 파일을 받는 서버의 주소
    #     processed_files = []

    #     progress_dialog = QProgressDialog("파일 다운로드 중...", None, 0, len(file_names), self)
    #     progress_dialog.setWindowTitle("다운로드 중")
    #     progress_dialog.setWindowModality(Qt.WindowModal)
        
    #     for file_name in file_names:
    #         try:
    #             response = requests.get(f"{url}/{file_name}")
    #             if response.status_code == 200:
    #                 save_path = os.path.join("/path/to/save/processed/files", file_name)  # 처리된 파일을 저장할 경로 설정
    #                 with open(save_path, 'wb') as f:
    #                     f.write(response.content)
    #                 processed_files.append(save_path)
    #             else:
    #                 QMessageBox.warning(self, "Failed", f"{file_name} 파일 다운로드 실패: {response.status_code}")
    #         except Exception as e:
    #             QMessageBox.critical(self, "Error", f"파일 다운로드 중 오류 발생: {str(e)}")

    #     self.model2.setStringList(processed_files)  # 처리된 파일 목록을 QStringListModel에 설정하여 QListView에 표시




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
        response = requests.post('http://192.168.0.201:5000/api/check', json=data)
        
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
        response = requests.post('http://192.168.0.201:5000/api/signup', json=data)
        
        
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
        response = requests.post('http://192.168.0.201:5000/api/signin', json=data)
        
        
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
    myWindows = WindowClass() 
    myWindows.show() 
    
    sys.exit(app.exec_()) 