import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic

from_class = uic.loadUiType("/home/ys/dev_ws/deep-repo-3/source/gui/Window.ui")[0]


class WindowClass(QMainWindow, from_class) :
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        self.setWindowTitle("Main Window") 
        
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
        
        
    def switch_to_windowPage(self):
        self.stackedWidget.setCurrentIndex(0)
        
    def switch_to_filesPage(self):
        self.stackedWidget.setCurrentIndex(1)
        
    def switch_to_detailsPage(self):
        self.stackedWidget.setCurrentIndex(2)
        
    def switch_to_settingsPage(self):
        self.stackedWidget.setCurrentIndex(4)
        
        
if __name__== "__main__":
    app = QApplication(sys.argv)
    myWindows = WindowClass()
    myWindows.show()
    
    sys.exit(app.exec_())