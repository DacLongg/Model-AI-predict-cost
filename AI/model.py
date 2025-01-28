from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from tensorflow import keras
from keras import Input
from datetime import datetime, timedelta
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from Get_cost import GET_COST
import threading
import time
import matplotlib.pyplot as plt

from Connect_UI import (Display_Cost_After_15M, Display_Cost_After_4H, Display_Cost_After_1H, Display_Acutal_value,
                        Display_Analysis_15M, Display_Analysis_4H, Display_Analysis_1H,
                        Display_Time_15M, Display_Time_1H, Display_Time_4H)

class Train_Model():
    def __init__(self):
        self.Data = None
        self.Scale = MinMaxScaler(feature_range=(0, 1))
        self.last_data_hash = None
        self.last_data_hash_in_loop = None

        self.model = Sequential([
            Input(shape=(60, 1)),  # Định nghĩa đầu vào ở đây
            LSTM(units=50, return_sequences=True),  # LSTM không cần `Input` nữa
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])

        self.model_30M = Sequential([
            Input(shape=(60, 1)),  # Định nghĩa đầu vào ở đây
            LSTM(units=50, return_sequences=True),  # LSTM không cần `Input` nữa
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])

        self.model_1H = Sequential([
            Input(shape=(60, 1)),  # Định nghĩa đầu vào ở đây
            LSTM(units=50, return_sequences=True),  # LSTM không cần `Input` nữa
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])

        # Compile mô hình
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model_30M.compile(optimizer='adam', loss='mean_squared_error')
        self.model_1H.compile(optimizer='adam', loss='mean_squared_error')

        self.First_No_Change = True
        self.mark_Actual = 1

        # Tạo figure cho biểu đồ

        self.scaler = MinMaxScaler(feature_range=(0, 1))

        self.predicted_values = []      # Lưu giá trị dự đoán
        self.actual_values = []         # Lưu giá trị thực tế
        self.predicted_value = 0
        self.Re_train_model_en = True

        self.Thread_Predict = threading.Thread(target=self.Predict_Loop)
        self.Stop_Thread_Predict = threading.Event()
        self.Thread_Predict.start()
        

    def Predict_Loop(self):
        while not self.Stop_Thread_Predict.is_set():
            try:
                # Lấy dữ liệu mới từ buffer
                current_buffer = GET_COST.Get_Buffer_15M()
                self.Value_Change = self.predicted_value - GET_COST.latest_price
                self.Percent_Change = (self.Value_Change * 100) / GET_COST.latest_price
                if self.Percent_Change > 0:
                    Display_Analysis_15M(f" Inc: {self.Percent_Change}%")
                else:
                    Display_Analysis_15M(f" Dec: {self.Percent_Change}%")

                # Kiểm tra nếu buffer rỗng hoặc không đủ dữ liệu
                if current_buffer is None or len(current_buffer) < 60:
                    print("Buffer không đủ dữ liệu. Đợi thêm...")
                    time.sleep(1)
                    continue

                # Kiểm tra nếu dữ liệu buffer không thay đổi
                current_hash = hash(tuple(current_buffer['close']))  # Hash dữ liệu cột 'close'
                if current_hash == self.last_data_hash_in_loop:
                    print("Dữ liệu không thay đổi. Bỏ qua dự đoán.")
                    time.sleep(1)
                    if self.First_No_Change != True:
                        continue
                    self.First_No_Change = False

                # Cập nhật hash của dữ liệu mới
                self.last_data_hash_in_loop = current_hash

                # Dự đoán giá trị tiếp theo
                self.predicted_value, self.actual_value = self.predict_and_evaluate(
                    current_buffer, self.model, self.scaler
                )

                # In kết quả dự đoán
                Display_Cost_After_15M(f"{self.predicted_value}")
                latest_timestamp = current_buffer['timestamp'].iloc[-1]  # Lấy giá trị thời gian mới nhất
                latest_timestamp = pd.to_datetime(latest_timestamp)  # Chuyển về dạng datetime
                updated_timestamp = latest_timestamp + timedelta(minutes=15)  # Cộng thêm 15 phút
                
                Display_Time_15M(updated_timestamp.strftime("%d %b %Y %H:%M:%S"))  # Hiển thị thời gian mới
                if self.actual_value != None:
                    Display_Acutal_value(f"{self.actual_value}")
                    self.mark_Actual = self.actual_value
                
                # print(f"Giá trị dự đoán: {self.predicted_value}")
                # print(f"Giá trị thực tế: {self.actual_value}")
                from Plot import PLOT_15M
                PLOT_15M.update_predicted(self.predicted_value)
                PLOT_15M.update_plot(self.actual_value)
                

                # Cập nhật dữ liệu kết quả để vẽ biểu đồ
                self.predicted_values.append(self.predicted_value)
                self.actual_values.append(self.actual_value)

                # Huấn luyện lại mô hình (tùy chọn)
                if len(current_buffer) >= 60 :
                    self.train_model_with_buffer(current_buffer, self.model, self.scaler)

            except Exception as e:
                print(f"Lỗi trong Predict_Loop: {e}")

            finally:
                time.sleep(3)
    
    def train_model_with_buffer(self, buffer, model, scaler, time_steps=60):
        """
        Huấn luyện hoặc cập nhật mô hình với dữ liệu từ bộ đệm.

        Args:
            buffer (deque): Bộ đệm dữ liệu.
            model: Mô hình LSTM.
            scaler: Bộ chuẩn hóa dữ liệu.
            time_steps (int): Số bước thời gian.

        Returns:
            None
        """
        # Chuyển buffer thành DataFrame (nếu buffer chứa DataFrame)
        if isinstance(buffer, pd.DataFrame):
            data = buffer['close'].values.reshape(-1, 1)  # Chỉ lấy cột 'close'
        else:
            # Nếu buffer không phải DataFrame, chuyển thành NumPy array
            data = np.array(buffer).reshape(-1, 1)

        # Chuẩn hóa dữ liệu
        scaled_data = scaler.fit_transform(data)

        # Tạo tập dữ liệu dạng supervised learning
        X, y = [], []
        for i in range(len(scaled_data) - time_steps):
            X.append(scaled_data[i:i + time_steps, 0])
            y.append(scaled_data[i + time_steps, 0])
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))              # Reshape thành dạng 3D

        model.fit(X, y, epochs=5, batch_size=32, verbose=1)     # Huấn luyện hoặc cập nhật mô hình

    def predict_and_evaluate(self, buffer, model, scaler, time_steps=60):
        """
        Dự đoán giá trị tiếp theo và kiểm tra tính chính xác khi có giá trị thực tế.
        Args:
            buffer (deque): Bộ đệm dữ liệu.
            model: Mô hình LSTM.
            scaler: Bộ chuẩn hóa dữ liệu.
            time_steps (int): Số bước thời gian.
        Returns:
            float: Giá trị dự đoán.
            float: Giá trị thực tế (nếu có).
        """
        
        if isinstance(buffer, pd.DataFrame):                    # Chuyển buffer thành DataFrame (nếu buffer chứa DataFrame)
            data = buffer['close'].values.reshape(-1, 1)        # Chỉ lấy cột 'close'
        else:                                                   # Nếu buffer không phải DataFrame, chuyển thành NumPy array
            data = np.array(buffer).reshape(-1, 1)
        if not hasattr(scaler, 'scale_'):                       # Kiểm tra nếu scaler đã được fit, nếu chưa thì fit
            scaler.fit(data)                                                    # Fit scaler trên toàn bộ dữ liệu buffer

        scaled_data = scaler.transform(data)                                    # Chuẩn hóa dữ liệu
        
        input_data = scaled_data[-time_steps:].reshape(1, -1, 1)                # Lấy dữ liệu đầu vào từ 60 giá trị gần nhất  
        predicted_scaled = model.predict(input_data)                            # Dự đoán giá trị tiếp theo
        predicted_value = scaler.inverse_transform(predicted_scaled)[0][0]

        
        actual_value = data[-1][0]                                  # Lấy giá trị thực tế (nếu có)
        current_hash = hash(tuple(buffer['close']))                 # Hash dữ liệu cột 'close'
        if current_hash == self.last_data_hash:
            print("Dữ liệu không thay đổi. Bỏ qua dự đoán.")
            actual_value = None
        
        self.last_data_hash = current_hash                          # Cập nhật hash của dữ liệu mới
        return predicted_value, actual_value
    def Stop(self):
        self.Stop_Thread_Predict.set()
        
        self.Thread_Predict.join(timeout=10)
        if self.Thread_Predict.is_alive():
            print("Stop Get cost fail")
        else:
            print("Stop Get cost Success")


class Train_Model_4H():
    def __init__(self):
        self.Data = None
        self.Scale = MinMaxScaler(feature_range=(0, 1))
        self.last_data_hash = None
        self.last_data_hash_in_loop = None

        self.model_4H = Sequential([
            Input(shape=(60, 1)),  # Định nghĩa đầu vào ở đây
            LSTM(units=50, return_sequences=True),  # LSTM không cần `Input` nữa
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])

        # Compile mô hình
        self.model_4H.compile(optimizer='adam', loss='mean_squared_error')
        self.First_No_Change = True

        self.scaler = MinMaxScaler(feature_range=(0, 1))

        self.predicted_values_4H = []      # Lưu giá trị dự đoán
        self.actual_values_4H = []         # Lưu giá trị thực tế

        self.predicted_value = 0
        self.Re_train_model_en = True

        self.Thread_Predict = threading.Thread(target=self.Predict_Loop)
        self.Stop_Thread_Predict = threading.Event()
        self.Thread_Predict.start()
        

    def Predict_Loop(self):
        while not self.Stop_Thread_Predict.is_set():
            try:
                # Lấy dữ liệu mới từ buffer
                current_buffer_4H = GET_COST.Get_Buffer_4H()
                self.Value_Change = self.predicted_value - GET_COST.latest_price
                self.Percent_Change = (self.Value_Change * 100) / GET_COST.latest_price
                if self.Percent_Change > 0:
                    Display_Analysis_4H(f" Inc: {self.Percent_Change}%")
                else:
                    Display_Analysis_4H(f" Dec: {self.Percent_Change}%")

                # Kiểm tra nếu buffer rỗng hoặc không đủ dữ liệu
                if current_buffer_4H is None or len(current_buffer_4H) < 60:
                    print("Buffer không đủ dữ liệu. Đợi thêm...")
                    time.sleep(1)
                    continue

                # Kiểm tra nếu dữ liệu buffer không thay đổi
                current_hash = hash(tuple(current_buffer_4H['close']))  # Hash dữ liệu cột 'close'
                if current_hash == self.last_data_hash_in_loop:
                    print("Dữ liệu không thay đổi. Bỏ qua dự đoán.")
                    time.sleep(1)
                    if self.First_No_Change != True:
                        continue
                    self.First_No_Change = False

                # Cập nhật hash của dữ liệu mới
                self.last_data_hash_in_loop = current_hash

                # Dự đoán giá trị tiếp theo
                self.predicted_value, self.actual_value = self.predict_and_evaluate(
                    current_buffer_4H, self.model_4H, self.scaler
                )

                # In kết quả dự đoán
                Display_Cost_After_4H(f"{self.predicted_value}")
                latest_timestamp = current_buffer_4H['timestamp'].iloc[-1]  # Lấy giá trị thời gian mới nhất
                latest_timestamp = pd.to_datetime(latest_timestamp)  # Chuyển về dạng datetime
                updated_timestamp = latest_timestamp + timedelta(hours=4)  # Cộng thêm 4h
                Display_Time_4H(updated_timestamp)
                if self.actual_value != None:
                    Display_Acutal_value(f"{self.actual_value}")
                    self.mark_Actual = self.actual_value
                
                # print(f"Giá trị dự đoán: {self.predicted_value}")
                # print(f"Giá trị thực tế: {self.actual_value}")
                from Plot import PLOT_4H
                PLOT_4H.update_predicted(self.predicted_value)
                PLOT_4H.update_plot(self.actual_value)
                

                # Cập nhật dữ liệu kết quả để vẽ biểu đồ
                self.predicted_values_4H.append(self.predicted_value)
                self.actual_values_4H.append(self.actual_value)

                # Huấn luyện lại mô hình (tùy chọn)
                if len(current_buffer_4H) >= 60:
                    self.train_model_with_buffer(current_buffer_4H, self.model_4H, self.scaler)
                    
            except Exception as e:
                print(f"Lỗi trong Predict_Loop: {e}")

            finally:
                # Chờ một khoảng thời gian trước khi kiểm tra lại
                time.sleep(10)
    
    def train_model_with_buffer(self, buffer, model, scaler, time_steps=60):
        """
        Huấn luyện hoặc cập nhật mô hình với dữ liệu từ bộ đệm.

        Args:
            buffer (deque): Bộ đệm dữ liệu.
            model: Mô hình LSTM.
            scaler: Bộ chuẩn hóa dữ liệu.
            time_steps (int): Số bước thời gian.

        Returns:
            None
        """
        # Chuyển buffer thành DataFrame (nếu buffer chứa DataFrame)
        if isinstance(buffer, pd.DataFrame):
            data = buffer['close'].values.reshape(-1, 1)  # Chỉ lấy cột 'close'
        else:
            # Nếu buffer không phải DataFrame, chuyển thành NumPy array
            data = np.array(buffer).reshape(-1, 1)

        # Chuẩn hóa dữ liệu
        scaled_data = scaler.fit_transform(data)

        # Tạo tập dữ liệu dạng supervised learning
        X, y = [], []
        for i in range(len(scaled_data) - time_steps):
            X.append(scaled_data[i:i + time_steps, 0])
            y.append(scaled_data[i + time_steps, 0])
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape thành dạng 3D

        # Huấn luyện hoặc cập nhật mô hình
        model.fit(X, y, epochs=5, batch_size=32, verbose=1)

    def predict_and_evaluate(self, buffer, model, scaler, time_steps=60):
        """
        Dự đoán giá trị tiếp theo và kiểm tra tính chính xác khi có giá trị thực tế.

        Args:
            buffer (deque): Bộ đệm dữ liệu.
            model: Mô hình LSTM.
            scaler: Bộ chuẩn hóa dữ liệu.
            time_steps (int): Số bước thời gian.

        Returns:
            float: Giá trị dự đoán.
            float: Giá trị thực tế (nếu có).
        """
        # Chuyển buffer thành DataFrame (nếu buffer chứa DataFrame)
        if isinstance(buffer, pd.DataFrame):
            data = buffer['close'].values.reshape(-1, 1)  # Chỉ lấy cột 'close'
        else:
            # Nếu buffer không phải DataFrame, chuyển thành NumPy array
            data = np.array(buffer).reshape(-1, 1)
        # Kiểm tra nếu scaler đã được fit, nếu chưa thì fit
        if not hasattr(scaler, 'scale_'):
            scaler.fit(data)  # Fit scaler trên toàn bộ dữ liệu buffer
        # Chuẩn hóa dữ liệu
        scaled_data = scaler.transform(data)
        # Lấy dữ liệu đầu vào từ 60 giá trị gần nhất
        input_data = scaled_data[-time_steps:].reshape(1, -1, 1)
        # Dự đoán giá trị tiếp theo
        predicted_scaled = model.predict(input_data)
        predicted_value = scaler.inverse_transform(predicted_scaled)[0][0]
        # Lấy giá trị thực tế (nếu có)
        actual_value = data[-1][0]

        current_hash = hash(tuple(buffer['close']))  # Hash dữ liệu cột 'close'
        if current_hash == self.last_data_hash:
            print("Dữ liệu không thay đổi. Bỏ qua dự đoán.")
            actual_value = None
        # Cập nhật hash của dữ liệu mới
        self.last_data_hash = current_hash
        return predicted_value, actual_value
    def Stop(self):
        self.Stop_Thread_Predict.set()
        
        self.Thread_Predict.join(timeout=10)
        if self.Thread_Predict.is_alive():
            print("Stop Get cost fail")
        else:
            print("Stop Get cost Success")

class Train_Model_1H():
    def __init__(self):
        self.Data = None
        self.Scale = MinMaxScaler(feature_range=(0, 1))
        self.last_data_hash = None
        self.last_data_hash_in_loop = None

        self.model_1H = Sequential([
            Input(shape=(60, 1)),  # Định nghĩa đầu vào ở đây
            LSTM(units=50, return_sequences=True),  # LSTM không cần `Input` nữa
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])

        # Compile mô hình
        self.model_1H.compile(optimizer='adam', loss='mean_squared_error')

        self.First_No_Change = True

        # Tạo figure cho biểu đồ

        self.scaler = MinMaxScaler(feature_range=(0, 1))

        self.predicted_values_1H = []      # Lưu giá trị dự đoán
        self.actual_values_1H = []         # Lưu giá trị thực tế
        self.predicted_value = 0
        self.Re_train_model_en = True

        self.Thread_Predict = threading.Thread(target=self.Predict_Loop)
        self.Stop_Thread_Predict = threading.Event()
        self.Thread_Predict.start()
        

    def Predict_Loop(self):
        while not self.Stop_Thread_Predict.is_set():
            try:
                # Lấy dữ liệu mới từ buffer
                current_buffer_1H  = GET_COST.Get_Buffer_1H()

                self.Value_Change = self.predicted_value - GET_COST.latest_price
                self.Percent_Change = (self.Value_Change * 100) / GET_COST.latest_price
                if self.Percent_Change > 0:
                    Display_Analysis_1H(f" Inc: {self.Percent_Change}%")
                else:
                    Display_Analysis_1H(f" Dec: {self.Percent_Change}%")


                # Kiểm tra nếu buffer rỗng hoặc không đủ dữ liệu
                if current_buffer_1H is None or len(current_buffer_1H) < 60:
                    print("Buffer không đủ dữ liệu. Đợi thêm...")
                    time.sleep(1)
                    continue

                # Kiểm tra nếu dữ liệu buffer không thay đổi
                current_hash = hash(tuple(current_buffer_1H['close']))  # Hash dữ liệu cột 'close'
                if current_hash == self.last_data_hash_in_loop:
                    print("Dữ liệu không thay đổi. Bỏ qua dự đoán.")
                    time.sleep(1)
                    if self.First_No_Change != True:
                        continue
                    self.First_No_Change = False

                # Cập nhật hash của dữ liệu mới
                self.last_data_hash_in_loop = current_hash

                # Dự đoán giá trị tiếp theo
                self.predicted_value, self.actual_value = self.predict_and_evaluate(
                    current_buffer_1H, self.model_1H, self.scaler
                )

                # In kết quả dự đoán
                Display_Cost_After_1H(f"{self.predicted_value}")
                latest_timestamp = current_buffer_1H['timestamp'].iloc[-1]  # Lấy giá trị thời gian mới nhất
                latest_timestamp = pd.to_datetime(latest_timestamp)  # Chuyển về dạng datetime
                updated_timestamp = latest_timestamp + timedelta(hours=1)  # Cộng thêm 1h
                Display_Time_1H(updated_timestamp)
                if self.actual_value != None:
                    Display_Acutal_value(f"{self.actual_value}")
                    self.mark_Actual = self.actual_value
                
                # print(f"Giá trị dự đoán: {self.predicted_value}")
                # print(f"Giá trị thực tế: {self.actual_value}")
                from Plot import PLOT_1H
                PLOT_1H.update_predicted(self.predicted_value)
                PLOT_1H.update_plot(self.actual_value)
                

                # Cập nhật dữ liệu kết quả để vẽ biểu đồ
                self.predicted_values_1H.append(self.predicted_value)
                self.actual_values_1H.append(self.actual_value)

                # Huấn luyện lại mô hình (tùy chọn)
                if len(current_buffer_1H) >= 60 :
                    self.train_model_with_buffer(current_buffer_1H, self.model_1H, self.scaler)

            except Exception as e:
                print(f"Lỗi trong Predict_Loop: {e}")

            finally:
                # Chờ một khoảng thời gian trước khi kiểm tra lại
                time.sleep(5)

    
    def train_model_with_buffer(self, buffer, model, scaler, time_steps=60):
        """
        Huấn luyện hoặc cập nhật mô hình với dữ liệu từ bộ đệm.

        Args:
            buffer (deque): Bộ đệm dữ liệu.
            model: Mô hình LSTM.
            scaler: Bộ chuẩn hóa dữ liệu.
            time_steps (int): Số bước thời gian.

        Returns:
            None
        """
        # Chuyển buffer thành DataFrame (nếu buffer chứa DataFrame)
        if isinstance(buffer, pd.DataFrame):
            data = buffer['close'].values.reshape(-1, 1)  # Chỉ lấy cột 'close'
        else:
            # Nếu buffer không phải DataFrame, chuyển thành NumPy array
            data = np.array(buffer).reshape(-1, 1)
        # Chuẩn hóa dữ liệu
        scaled_data = scaler.fit_transform(data)

        # Tạo tập dữ liệu dạng supervised learning
        X, y = [], []
        for i in range(len(scaled_data) - time_steps):
            X.append(scaled_data[i:i + time_steps, 0])
            y.append(scaled_data[i + time_steps, 0])
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape thành dạng 3D

        # Huấn luyện hoặc cập nhật mô hình
        model.fit(X, y, epochs=5, batch_size=32, verbose=1)

    def predict_and_evaluate(self, buffer, model, scaler, time_steps=60):
        """
        Dự đoán giá trị tiếp theo và kiểm tra tính chính xác khi có giá trị thực tế.

        Args:
            buffer (deque): Bộ đệm dữ liệu.
            model: Mô hình LSTM.
            scaler: Bộ chuẩn hóa dữ liệu.
            time_steps (int): Số bước thời gian.

        Returns:
            float: Giá trị dự đoán.
            float: Giá trị thực tế (nếu có).
        """
        # Chuyển buffer thành DataFrame (nếu buffer chứa DataFrame)
        if isinstance(buffer, pd.DataFrame):
            data = buffer['close'].values.reshape(-1, 1)  # Chỉ lấy cột 'close'
        else:
            # Nếu buffer không phải DataFrame, chuyển thành NumPy array
            data = np.array(buffer).reshape(-1, 1)
        # Kiểm tra nếu scaler đã được fit, nếu chưa thì fit
        if not hasattr(scaler, 'scale_'):
            scaler.fit(data)  # Fit scaler trên toàn bộ dữ liệu buffer

        # Chuẩn hóa dữ liệu
        scaled_data = scaler.transform(data)

        # Lấy dữ liệu đầu vào từ 60 giá trị gần nhất
        input_data = scaled_data[-time_steps:].reshape(1, -1, 1)

        # Dự đoán giá trị tiếp theo
        predicted_scaled = model.predict(input_data)
        predicted_value = scaler.inverse_transform(predicted_scaled)[0][0]

        # Lấy giá trị thực tế (nếu có)
        actual_value = data[-1][0]

        current_hash = hash(tuple(buffer['close']))  # Hash dữ liệu cột 'close'
        if current_hash == self.last_data_hash:
            print("Dữ liệu không thay đổi. Bỏ qua dự đoán.")
            actual_value = None
        # Cập nhật hash của dữ liệu mới
        self.last_data_hash = current_hash
        return predicted_value, actual_value
    def Stop(self):
        self.Stop_Thread_Predict.set()
        
        self.Thread_Predict.join(timeout=10)
        if self.Thread_Predict.is_alive():
            print("Stop Get cost fail")
        else:
            print("Stop Get cost Success")

    
MODEL = Train_Model()
MODEL1 = Train_Model_1H()
MODEL2 = Train_Model_4H()
# data = GET_COST.Get_Cost_From_Queue()

# # Thiết lập 'timestamp' làm chỉ mục
# data.set_index('timestamp', inplace=True)

# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data = scaler.fit_transform(data[['close']])  # Chỉ chuẩn hóa cột giá đóng cửa

# print(scaled_data)

# # Tạo tập dữ liệu theo dạng supervised learning
# def create_dataset(data, time_steps=60):
#     X, y = [], []
#     for i in range(len(data) - time_steps):
#         X.append(data[i:i + time_steps, 0])  # 60 giá trị đóng cửa trước đó
#         y.append(data[i + time_steps, 0])   # Giá trị tiếp theo
#     return np.array(X), np.array(y)

# # Sử dụng 60 bước thời gian (60 giờ) để dự đoán giá tiếp theo
# time_steps = 60
# X, y = create_dataset(scaled_data, time_steps)

# # Reshape dữ liệu X thành dạng 3D (samples, timesteps, features) cho LSTM
# X = X.reshape((X.shape[0], X.shape[1], 1))



# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# # Khởi tạo mô hình
# model = Sequential([
#     Input(shape=(60, 1)),  # Định nghĩa đầu vào ở đây
#     LSTM(units=50, return_sequences=True),  # LSTM không cần `Input` nữa
#     Dropout(0.2),
#     LSTM(units=50, return_sequences=False),
#     Dropout(0.2),
#     Dense(25),
#     Dense(1)
# ])


# # Compile mô hình
# model.compile(optimizer='adam', loss='mean_squared_error')


# history = model.fit(
#     X_train, y_train,
#     batch_size=32,  # Kích thước batch
#     epochs=50,      # Số lần lặp
#     validation_data=(X_test, y_test),
#     verbose=1
# )


# # Dự đoán
# predicted_prices = model.predict(X_test)
# predicted_prices = scaler.inverse_transform(predicted_prices)  # Chuyển về giá trị ban đầu

# # Giá trị thực tế
# true_prices = scaler.inverse_transform(y_test.reshape(-1, 1))


# mse = mean_squared_error(true_prices, predicted_prices)
# rmse = np.sqrt(mse)
# print("RMSE:", rmse)

# # Lấy giá đóng cửa mới nhất
# new_data = scaled_data[-time_steps:]  # Lấy 60 giá trị đóng cửa gần nhất
# new_data = new_data.reshape(1, -1, 1)  # Reshape thành 3D cho LSTM

# predicted_price = model.predict(new_data)
# predicted_price = scaler.inverse_transform(predicted_price)
# print("Giá dự đoán tiếp theo:", predicted_price[0][0])




# plt.figure(figsize=(10, 6))
# plt.plot(true_prices, color='blue', label='Giá thực tế')
# plt.plot(predicted_prices, color='red', label='Giá dự đoán')
# plt.title('Dự đoán giá coin')
# plt.xlabel('Thời gian')
# plt.ylabel('Giá')
# plt.legend()
# plt.show()
