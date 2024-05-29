import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
# import res2_rc
from PyQt5.QtCore import Qt
import requests

from_class = uic.loadUiType('/home/hb/dev_ws/running/deep/project/Login/login.ui')[0] # 현복 login ui파일 
sign_up_class = uic.loadUiType('/home/hb/dev_ws/running/deep/project/Login/signup.ui')[0] # 현복 Sign_up ui 파일

total_gui_class = uic.loadUiType('/home/hb/dev_ws/running/deep/project/Login/Window.ui')[0] # 영수님 Main_ui파일

# Total GUI 화면 클래스 (영수님 mainwindow file 복붙 ㄱㄱ)
class Total_gui_Window(QMainWindow, total_gui_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        self.setWindowTitle("Driving Test") 
        
        self.stacked_widget = self.findChild(QStackedWidget, 'stackedWidget')
        
        self.sign1.clicked.connect(self.switch_login) # Logout 버튼
        self.sign.clicked.connect(self.switch_login) # Logout 버튼
        
        self.pushButton_3.clicked.connect(self.switch_detail) # detail page 전환
        
        
        self.windows.clicked.connect(self.switch_to_windowPage)
        self.windows1.clicked.connect(self.switch_to_windowPage)
        
        self.files.clicked.connect(self.switch_to_filesPage)
        self.files1.clicked.connect(self.switch_to_filesPage)
        
        self.details.clicked.connect(self.switch_to_detailsPage)
        self.details1.clicked.connect(self.switch_to_detailsPage)
        
        self.set.clicked.connect(self.switch_to_settingsPage)
        self.set1.clicked.connect(self.switch_to_settingsPage)
        
        self.model = QFileSystemModel()
        self.model.setRootPath('')

        self.fileTreeView.setModel(self.model)
        self.fileTreeView.setRootIndex(self.model.index(''))
        self.fileTreeView.setColumnWidth(0, 250)
    
    
    # Detail 화면 전환
    def switch_detail(self):
        self.stacked_widget.setCurrentIndex(2)
    
    
    # Logout 기능
    def switch_login(self):
        reply = QMessageBox.question(self, 'Message', '로그인 화면으로 돌아가시겠습니까?', 
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.go_back()
            
        else:
            pass
    
    def go_back(self):
        self.close()
        self.previous_window = WindowClass()
        self.previous_window.show()
    
    
    def switch_to_windowPage(self):
        self.stackedWidget.setCurrentIndex(0)
        
    def switch_to_filesPage(self):
        self.stackedWidget.setCurrentIndex(1)
        
    def switch_to_detailsPage(self):
        self.stackedWidget.setCurrentIndex(2)
        
    def switch_to_settingsPage(self):
        self.stackedWidget.setCurrentIndex(4)
        



# Sign up Class
class Sign_up_Window(QDialog, sign_up_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.Sign.clicked.connect(self.sign_up)
        self.Check.clicked.connect(self.Check_id)
        
        
        self.setFixedSize(self.size())
        
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

        self.setFixedSize(self.size())

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