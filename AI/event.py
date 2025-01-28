from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QCloseEvent, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QGraphicsView,
    QGroupBox, QHeaderView, QLabel, QLineEdit,
    QProgressBar, QPushButton, QSizePolicy, QTableWidget,
    QTableWidgetItem, QTextEdit, QWidget)
from PySide6.QtWidgets import QMainWindow
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QFileDialog, QMessageBox

from ui_form import Ui_MainWindow
from Connect_UI import Init_Object
import ctypes
import pyqtgraph as pg
from Get_cost import GET_COST
from model import  MODEL2 , MODEL, MODEL1

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setupConnections()

        Init_Object(self.ui)
        self.Init_Graph()
        from Plot import Cofig_Plot
        Cofig_Plot(self.ui, 
                   self.Actuare_curve_15M, self.predicted_curve_15M,      
                   self.Actuare_curve_1H, self.predicted_curve_1H,
                   self.Actuare_curve_4H, self.predicted_curve_4H
                   )

    def setupConnections(self):                             # hàm kết nối sự kiện button
        print("connrct event")
        # self.ui.pushButton_Read.clicked.connect(self.ReadData)
    def Init_Graph(self):
        self.plot_widget_15M = pg.PlotWidget()
        self.ui.verticalLayout_Predict_15M.addWidget(self.plot_widget_15M)

        self.plot_widget_1H = pg.PlotWidget()
        self.ui.verticalLayout_Predict_1H.addWidget(self.plot_widget_1H)

        self.plot_widget_4H = pg.PlotWidget()
        self.ui.verticalLayout_Predict_4H.addWidget(self.plot_widget_4H)

        # Cài đặt trục
        self.plot_widget_15M.setLabel('bottom', "Time")
        self.plot_widget_15M.setLabel('left', "Cost")
        self.plot_widget_15M.addLegend()

        self.plot_widget_1H.setLabel('bottom', "Time")
        self.plot_widget_1H.setLabel('left', "Cost")
        self.plot_widget_1H.addLegend()

        self.plot_widget_4H.setLabel('bottom', "Time")
        self.plot_widget_4H.setLabel('left', "Cost")
        self.plot_widget_4H.addLegend()


        self.x_data = []
        self.y_data = []
        self.x_data_predicted = []  # Dữ liệu trục x cho giá trị dự đoán
        self.y_data_predicted = []  # Dữ liệu trục y cho giá trị dự đoán
        # Dòng dữ liệu
        self.Actuare_curve_15M = self.plot_widget_15M.plot(
            self.x_data, 
            self.y_data, 
            pen=pg.mkPen(color='b', width=2), 
            symbol='o', 
            symbolSize=8, 
            symbolBrush='r', 
            name="Giá trị"
        )
        self.predicted_curve_15M = self.plot_widget_15M.plot(
            self.x_data_predicted, 
            self.y_data_predicted, 
            pen=pg.mkPen(color='r', width=2, style=pg.QtCore.Qt.DashLine), 
            symbol='o', 
            symbolSize=8, 
            symbolBrush='r', 
            name="Dự đoán"
        )

        self.Actuare_curve_4H = self.plot_widget_4H.plot(
            self.x_data, 
            self.y_data, 
            pen=pg.mkPen(color='b', width=2), 
            symbol='o', 
            symbolSize=8, 
            symbolBrush='r', 
            name="Giá trị"
        )
        self.predicted_curve_4H = self.plot_widget_4H.plot(
            self.x_data_predicted, 
            self.y_data_predicted, 
            pen=pg.mkPen(color='r', width=2, style=pg.QtCore.Qt.DashLine), 
            symbol='o', 
            symbolSize=8, 
            symbolBrush='r', 
            name="Dự đoán"
        )

        self.Actuare_curve_1H = self.plot_widget_1H.plot(
            self.x_data, 
            self.y_data, 
            pen=pg.mkPen(color='b', width=2), 
            symbol='o', 
            symbolSize=8, 
            symbolBrush='r', 
            name="Giá trị"
        )
        self.predicted_curve_1H = self.plot_widget_1H.plot(
            self.x_data_predicted, 
            self.y_data_predicted, 
            pen=pg.mkPen(color='r', width=2, style=pg.QtCore.Qt.DashLine), 
            symbol='o', 
            symbolSize=8, 
            symbolBrush='r', 
            name="Dự đoán"
        )
    
    def Display_table(self):
        pass

    def closeEvent(self, event: QCloseEvent) -> None:
        print(" close")
        GET_COST.Stop()
        MODEL.Stop()
        MODEL1.Stop()
        MODEL2.Stop()

        
        return super().closeEvent(event)