from PySide6 import QtWidgets
import pyqtgraph as pg
import sys
import threading
import time
from datetime import datetime, timedelta

class LivePlot():
    def __init__(self, Window, Acture_curve, Predict_curve):
        self.Ui = Window
        self.Acture_curve = Acture_curve
        self.predicted_curve = Predict_curve

        self.max_points = 20
        # Dữ liệu mặc định
        self.x_data = []
        self.y_data = []

        self.x_data_predicted = []
        self.y_data_predicted = []

        self.new_x_acture = 0
        self.new_x_predict = 0

    def update_plot(self, new_y):
        """
        Hàm cập nhật dữ liệu vào đồ thị.
        :param new_x: Danh sách hoặc giá trị x mới.
        :param new_y: Danh sách hoặc giá trị y mới.
        """
        if new_y == None:
            return
        if len(self.x_data) == 0:
            # Nếu danh sách rỗng, bắt đầu từ 0 phút
            self.new_x_acture = 0
        else:
            # Thêm 15 phút so với giá trị cuối cùng
            self.new_x_acture = self.x_data[-1] + 15
        if isinstance(new_y, list):
            self.x_data.extend(self.new_x_acture)
            self.y_data.extend(new_y)
        else:
            self.x_data.append(self.new_x_acture)
            self.y_data.append(new_y)
        # Kiểm tra và giới hạn số lượng điểm
        if len(self.x_data) > self.max_points:
            self.x_data = self.x_data[-self.max_points:]
            self.y_data = self.y_data[-self.max_points:]
        # Cập nhật đồ thị
        self.Acture_curve.setData(self.x_data, self.y_data)
    def update_predicted(self, new_y):
        """
        Cập nhật dữ liệu cho đường dự đoán.
        :param new_x: Danh sách hoặc giá trị x mới.
        :param new_y: Danh sách hoặc giá trị y mới.
        """
        if new_y == None:
            return
        # Tự động tính new_x bằng cách thêm 15 phút
        if len(self.x_data_predicted) == 0:
            # Nếu danh sách rỗng, bắt đầu từ 0 phút
            self.new_x_predict = 0
        else:
            # Thêm 15 phút so với giá trị cuối cùng
            self.new_x_predict = self.x_data_predicted[-1] + 15
        if isinstance(new_y, list):
            self.x_data_predicted.extend(self.new_x_predict)
            self.y_data_predicted.extend(new_y)
        else:
            self.x_data_predicted.append(self.new_x_predict)
            self.y_data_predicted.append(new_y)
        # Kiểm tra và giới hạn số lượng điểm
        if len(self.x_data_predicted) > self.max_points:
            self.x_data_predicted = self.x_data_predicted[-self.max_points:]
            self.y_data_predicted = self.y_data_predicted[-self.max_points:]

        # Cập nhật đồ thị dự đoán
        self.predicted_curve.setData(self.x_data_predicted, self.y_data_predicted)

PLOT_15M = None
PLOT_1H = None
PLOT_4H = None

def Cofig_Plot(window, 
               Acture_curve_15M, Predict_curve_15M, 
               Acture_curve_1H, Predict_curve_1H, 
               Acture_curve_4H, Predict_curve_4H,):
    global PLOT_15M, PLOT_4H, PLOT_1H
    PLOT_15M = LivePlot(window, Acture_curve_15M, Predict_curve_15M)
    PLOT_1H = LivePlot(window, Acture_curve_1H, Predict_curve_1H)
    PLOT_4H  = LivePlot(window, Acture_curve_4H, Predict_curve_4H)