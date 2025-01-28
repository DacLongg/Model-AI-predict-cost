# This Python file uses the following encoding: utf-8
import sys

from PySide6.QtWidgets import QApplication, QWidget

# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic Bitcoin_cost.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
# from ui_form import Ui_miniTool

# pyside6-designer
from event import MainWindow



if __name__ == "__main__":
    app = QApplication.instance()
    if app is None:  # Nếu chưa có thì khởi tạo mới
        app = QApplication(sys.argv)
    # widget = miniTool()
    # widget.show()
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec())
