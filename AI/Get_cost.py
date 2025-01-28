from binance.client import Client
from binance import ThreadedWebsocketManager
import pandas as pd
import threading
import time
from datetime import datetime, timedelta
from collections import deque
from Connect_UI import Display_Time_now, Write_Table, Display_Cost_Current
import pytz

class Get_Cost():
    def __init__(self):
        # Khởi tạo Binance API Client
        api_key = "your_api_key"
        api_secret = "your_api_secret"
        self.client = Client(api_key, api_secret)

         # WebSocket Manager
        self.twm = ThreadedWebsocketManager(api_key=api_key, api_secret=api_secret)
        self.twm.start()
        # Start WebSocket stream
        self.Start_WebSocket_Stream()

        self.CountTime = 900
        self.Cost_Symbol = None
        self.Buffer_History_Cost_15M = deque(maxlen=500)
        self.Buffer_History_Cost_1H = deque(maxlen=500)
        self.Buffer_History_Cost_4H  = deque(maxlen=500)

        self.merged_data_15M = [0]
        self.merged_data_1H = [0]
        self.merged_data_4H = [0]

        self.Current_time = datetime.now()
        self.Current_time = self.Current_time.astimezone(pytz.utc)
        self.Current_formatted_time = self.Current_time.strftime("%d %b %Y %H:%M:%S")

        self.time_before_15M = self.Current_time - timedelta(days=5)
        self.time_after_15M = self.time_before_15M + timedelta(hours=1)
        # Định dạng thời gian
        self.before_formatted_time_15M = self.time_before_15M.strftime("%d %b %Y %H:%M:%S")
        self.After_formartted_time_15M = self.time_after_15M.strftime("%d %b %Y %H:%M:%S")

        self.time_before_1H = self.Current_time - timedelta(days=20)
        self.time_after_1H = self.time_before_1H + timedelta(hours=4)
        # Định dạng thời gian
        self.before_formatted_time_1H = self.time_before_1H.strftime("%d %b %Y %H:%M:%S")
        self.After_formartted_time_1H = self.time_after_1H.strftime("%d %b %Y %H:%M:%S")

        self.time_before_4H = self.Current_time - timedelta(days=40)
        self.time_after_4H = self.time_before_4H + timedelta(hours=16)
        # Định dạng thời gian
        self.before_formatted_time_4H = self.time_before_4H.strftime("%d %b %Y %H:%M:%S")
        self.After_formartted_time_4H = self.time_after_4H.strftime("%d %b %Y %H:%M:%S")

        self.latest_Time_15M = self.time_before_15M
        self.latest_Time_1H = self.time_before_1H
        self.latest_Time_4H = self.time_before_4H

        self.Thread_get_cost = threading.Thread(target=self.Get_Cost_Handle)
        self.Stop_Thread_get_cost = threading.Event()
        self.Thread_get_cost.start()
        
    def Get_Cost_Handle(self):
        while not self.Stop_Thread_get_cost.is_set():
            self.Current_time = datetime.now()
            self.Current_time = self.Current_time.astimezone(pytz.utc)

            
            # Lấy danh sách các đồng coin
            all_coins = self.get_all_coins()
            print("Danh sách các đồng coin trên Binance:")
            print(all_coins)
            
            if len(self.merged_data_15M) < 60 or self.latest_Time_15M < self.Current_time - timedelta(hours=4):
                self.time_after_15M = self.latest_Time_15M + timedelta(hours=4)
                self.before_formatted_time_15M = self.latest_Time_15M.strftime("%d %b %Y %H:%M:%S")
                self.After_formartted_time_15M = self.time_after_15M.strftime("%d %b %Y %H:%M:%S")

                self.Cost_Symbol = self.get_historical_data("ETHUSDT", Client.KLINE_INTERVAL_15MINUTE, self.before_formatted_time_15M, self.After_formartted_time_15M)
                self.time_before_15M = self.time_after_15M

                self.Buffer_History_Cost_15M.append(self.Cost_Symbol)
            elif self.latest_Time_15M >= self.Current_time - timedelta(hours=4) and self.latest_Time_15M <= self.Current_time - timedelta(minutes=15):
                self.time_after_15M = self.latest_Time_15M + timedelta(minutes=15)
                self.before_formatted_time_15M = self.latest_Time_15M.strftime("%d %b %Y %H:%M:%S")
                self.After_formartted_time_15M = self.time_after_15M.strftime("%d %b %Y %H:%M:%S")

                self.Cost_Symbol = self.get_historical_data("ETHUSDT", Client.KLINE_INTERVAL_15MINUTE, self.before_formatted_time_15M, self.After_formartted_time_15M)
                self.time_before_15M = self.time_after_15M
                self.Buffer_History_Cost_15M.append(self.Cost_Symbol)

            # Get cost with delta t = 1h
            if len(self.merged_data_1H) < 60 or self.latest_Time_1H < self.Current_time - timedelta(hours=8):
                self.time_after_1H = self.latest_Time_1H + timedelta(hours=8)
                self.before_formatted_time_1H = self.latest_Time_1H.strftime("%d %b %Y %H:%M:%S")
                self.After_formartted_time_1H = self.time_after_1H.strftime("%d %b %Y %H:%M:%S")

                self.Cost_Symbol = self.get_historical_data("ETHUSDT", Client.KLINE_INTERVAL_1HOUR, self.before_formatted_time_1H, self.After_formartted_time_1H)
                self.time_before_1H = self.time_after_1H

                self.Buffer_History_Cost_1H.append(self.Cost_Symbol)
            elif self.latest_Time_1H > self.Current_time - timedelta(hours=8) and self.latest_Time_1H <= self.Current_time - timedelta(hours=1):
                self.time_after_1H = self.latest_Time_1H + timedelta(hours=1)
                self.before_formatted_time_1H = self.latest_Time_1H.strftime("%d %b %Y %H:%M:%S")
                self.After_formartted_time_1H = self.time_after_1H.strftime("%d %b %Y %H:%M:%S")

                self.Cost_Symbol = self.get_historical_data("ETHUSDT", Client.KLINE_INTERVAL_1HOUR, self.before_formatted_time_1H, self.After_formartted_time_1H)
                self.time_before_1H = self.time_after_1H
                self.Buffer_History_Cost_1H.append(self.Cost_Symbol)

            # Get cost with delta t = 4h
            if len(self.merged_data_4H) < 60 or self.latest_Time_4H < self.Current_time - timedelta(hours=16):
                self.time_after_4H = self.latest_Time_4H + timedelta(hours=16)
                self.before_formatted_time_4H = self.latest_Time_4H.strftime("%d %b %Y %H:%M:%S")
                self.After_formartted_time_4H = self.time_after_4H.strftime("%d %b %Y %H:%M:%S")

                self.Cost_Symbol = self.get_historical_data("ETHUSDT", Client.KLINE_INTERVAL_4HOUR, self.before_formatted_time_4H, self.After_formartted_time_4H)
                self.time_before_4H = self.time_after_4H

                self.Buffer_History_Cost_4H.append(self.Cost_Symbol)
            elif self.latest_Time_4H > self.Current_time - timedelta(hours=16) and self.latest_Time_4H < self.Current_time - timedelta(hours=4):
                self.time_after_4H = self.latest_Time_4H + timedelta(hours=4)
                self.before_formatted_time_4H = self.latest_Time_4H.strftime("%d %b %Y %H:%M:%S")
                self.After_formartted_time_4H = self.time_after_4H.strftime("%d %b %Y %H:%M:%S")

                self.Cost_Symbol = self.get_historical_data("ETHUSDT", Client.KLINE_INTERVAL_4HOUR, self.before_formatted_time_4H, self.After_formartted_time_4H)
                self.time_before_4H = self.time_after_4H
                self.Buffer_History_Cost_4H.append(self.Cost_Symbol)
            

            time.sleep(1)
            if self.Stop_Thread_get_cost.is_set():
                break
    def get_historical_data(self, symbol, interval, start, end):
        candles = self.client.get_historical_klines(
            symbol=symbol, 
            interval=interval, 
            start_str=start, 
            end_str=end
        )
        # Tạo DataFrame
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 
                'taker_buy_quote_asset_volume', 'ignore']
        df = pd.DataFrame(candles, columns=columns)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  # Định dạng thời gian
        Data = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        Data.loc[:, 'open'] = Data['open'].astype(float)
        Data.loc[:, 'high'] = Data['high'].astype(float)
        Data.loc[:, 'low'] = Data['low'].astype(float)
        Data.loc[:, 'close'] = Data['close'].astype(float)
        Data.loc[:, 'volume'] = Data['volume'].astype(float)

        Data = Data[['timestamp', 'close']]
        # print(Data)
        return Data
    def Get_Buffer_15M(self):
        
        # Chuyển deque thành danh sách
        data_list = list(self.Buffer_History_Cost_15M)

        # Gộp dữ liệu thành DataFrame lớn
        if data_list:
            self.merged_data_15M = pd.concat(data_list, ignore_index=True)
            self.merged_data_15M['timestamp'] = self.merged_data_15M['timestamp'].dt.strftime("%d %b %Y %H:%M:%S")
            if len(self.merged_data_15M) > 1000:
                excess_rows = len(self.merged_data_15M) - 1000
                self.merged_data_15M = self.merged_data_15M.iloc[excess_rows:].reset_index(drop=True)  # Xóa các dòng đầu và reset index
            print("Dữ liệu với D = 15M:, len = ", len(self.merged_data_15M))
            print(self.merged_data_15M)

            self.latest_Time_15M = self.merged_data_15M['timestamp'].iloc[-1]  # Lấy giá trị thời gian mới nhất
            self.latest_Time_15M = pd.to_datetime(self.latest_Time_15M, utc=True)
            return self.merged_data_15M
        else:
            print("Deque rỗng.")
            return None
    def Get_Buffer_1H(self):
        
        # Chuyển deque thành danh sách
        data_list = list(self.Buffer_History_Cost_1H)

        # Gộp dữ liệu thành DataFrame lớn
        if data_list:
            self.merged_data_1H = pd.concat(data_list, ignore_index=True)
            self.merged_data_1H['timestamp'] = self.merged_data_1H['timestamp'].dt.strftime("%d %b %Y %H:%M:%S")
            
            # Kiểm tra và giới hạn độ dài 1000
            if len(self.merged_data_1H) > 1000:
                excess_rows = len(self.merged_data_1H) - 1000
                self.merged_data_1H = self.merged_data_1H.iloc[excess_rows:].reset_index(drop=True)  # Xóa các dòng đầu và reset index
            print("Dữ liệu D = 1H", len(self.merged_data_1H))
            print(self.merged_data_1H)

            self.latest_Time_1H = self.merged_data_1H['timestamp'].iloc[-1]  # Lấy giá trị thời gian mới nhất
            self.latest_Time_1H = pd.to_datetime(self.latest_Time_1H, utc=True)
            return self.merged_data_1H
        else:
            print("Deque rỗng.")
            return None
    def Get_Buffer_4H(self):
        
        # Chuyển deque thành danh sách
        data_list = list(self.Buffer_History_Cost_4H)

        # Gộp dữ liệu thành DataFrame lớn
        if data_list:
            self.merged_data_4H = pd.concat(data_list, ignore_index=True)
            self.merged_data_4H['timestamp'] = self.merged_data_4H['timestamp'].dt.strftime("%d %b %Y %H:%M:%S")
            if len(self.merged_data_4H) > 1000:
                excess_rows = len(self.merged_data_4H) - 1000
                self.merged_data_4H = self.merged_data_4H.iloc[excess_rows:].reset_index(drop=True)  # Xóa các dòng đầu và reset index
            print("Dữ liệu D = 4H", len(self.merged_data_4H))
            print(self.merged_data_4H)

            self.latest_Time_4H = self.merged_data_4H['timestamp'].iloc[-1]  # Lấy giá trị thời gian mới nhất
            self.latest_Time_4H = pd.to_datetime(self.latest_Time_4H, utc=True)
            return self.merged_data_4H
        else:
            print("Deque rỗng.")
            return None
            
    def Start_WebSocket_Stream(self):
        """
        Khởi chạy WebSocket stream để lắng nghe giá BTC/USDT mới nhất.
        """
        self.twm.start_symbol_ticker_socket(
            callback=self.WebSocket_Callback,
            symbol="ETHUSDT"
        )
        print("WebSocket stream started for BTCUSDT.")

    def WebSocket_Callback(self, msg):
        """
        Callback xử lý dữ liệu từ WebSocket.
        """
        if 'c' in msg:  # 'c' là giá đóng cửa hiện tại từ WebSocket
            self.latest_price = float(msg['c'])
            # print(f"Giá BTCUSDT mới nhất (WebSocket): {self.latest_price} USDT")
            Display_Cost_Current(self.latest_price)
            Display_Time_now(self.Current_formatted_time)

    def get_all_coins(self):
        # Lấy thông tin toàn bộ các cặp giao dịch
        exchange_info = self.client.get_exchange_info()
        symbols = exchange_info['symbols']
        
        # Trích xuất danh sách các đồng coin (baseAsset)
        coins = set([symbol['baseAsset'] for symbol in symbols])
        return sorted(coins)  # Sắp xếp theo thứ tự alphabet

    def Stop(self):
        self.Stop_Thread_get_cost.set()
        
        self.Thread_get_cost.join(timeout=10)
        if self.Thread_get_cost.is_alive():
            print("Stop Get cost fail")
        else:
            print("Stop Get cost Success")
            
    
GET_COST = Get_Cost()