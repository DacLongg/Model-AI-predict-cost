
from PySide6.QtWidgets import QTableWidgetItem
Label_15M = None
Label_4H = None
Label_1H  = None
Label_Now = None
Line_ED_Cur = None
Line_ED_15M = None
Line_ED_4H = None
Line_ED_1H  = None
Line_ED_Actual = None

Label_ANALYSIS_15M = None
Label_ANALYSIS_4H = None
Label_ANALYSIS_1H  = None

Label_Time_15M = None
Label_Time_4H = None
Label_Time_1H  = None

Table = None

def Init_Object(Ui):
    global Line_ED_15M, Line_ED_4H, Line_ED_1H, Label_Now, Table, Line_ED_Cur, Line_ED_Actual, Label_ANALYSIS_15M, Label_ANALYSIS_4H, Label_ANALYSIS_1H
    global Label_Time_15M, Label_Time_1H, Label_Time_4H
    Line_ED_15M = Ui.lineEdit_15M
    Line_ED_4H = Ui.lineEdit_4H
    Line_ED_1H  = Ui.lineEdit_1H
    Line_ED_Cur = Ui.lineEdit_Current_cost
    Label_Now   = Ui.label_TimeNow
    Line_ED_Actual = Ui.lineEdit_Acture_value
    Label_ANALYSIS_15M = Ui.label_Result_15M
    Label_ANALYSIS_4H = Ui.label_Result_4H
    Label_ANALYSIS_1H  = Ui.label_Result_1H
    Label_Time_15M = Ui.label_Analysis_15M
    Label_Time_1H = Ui.label_Analysis_1H
    Label_Time_4H = Ui.label_Analysis_4H
    # Table       = Ui.tableWidget

def Display_Time_15M(Value):
    if Label_Time_15M != None:
        if Value != str:
            Label_Time_15M.setText(str(Value))
        else:
            Label_Time_15M.setText(Value)

def Display_Time_1H(Value):
    if Label_Time_1H != None:
        if Value != str:
            Label_Time_1H.setText(str(Value))
        else:
            Label_Time_1H.setText(Value)

def Display_Time_4H(Value):
    if Label_Time_4H != None:
        if Value != str:
            Label_Time_4H.setText(str(Value))
        else:
            Label_Time_4H.setText(Value)

def Display_Analysis_15M(Value):
    if Label_ANALYSIS_15M != None:
        if Value != str:
            Label_ANALYSIS_15M.setText(str(Value))
        else:
            Label_ANALYSIS_15M.setText(Value)

def Display_Analysis_4H(Value):
    if Label_ANALYSIS_4H != None:
        if Value != str:
            Label_ANALYSIS_4H.setText(str(Value))
        else:
            Label_ANALYSIS_4H.setText(Value)

def Display_Analysis_1H(Value):
    if Label_ANALYSIS_1H != None:
        if Value != str:
            Label_ANALYSIS_1H.setText(str(Value))
        else:
            Label_ANALYSIS_1H.setText(Value)

def Display_Acutal_value(Value):
    if Line_ED_Actual != None:
        if Value != str:
            Line_ED_Actual.setText(str(Value))
        else:
            Line_ED_Actual.setText(Value)

def Display_Cost_Current(Value):
    if Line_ED_Cur != None:
        if Value != str:
            Line_ED_Cur.setText(str(Value))
        else:
            Line_ED_Cur.setText(Value)

def Display_Cost_After_15M(Value):
    Line_ED_15M.setText(str(Value))

def Display_Cost_After_4H(Value):
    Line_ED_4H.setText(str(Value))

def Display_Cost_After_1H(Value):
    Line_ED_1H.setText(str(Value))

def Display_Time_now(Time):
    if Label_Now != None:
        Label_Now.setText(str(Time))

def Write_Table(row_label, column_label, value):
    """
    Cập nhật giá trị vào một ô dựa trên tên hàng và tên cột.

    Args:
        Table (QTableWidget): Đối tượng QTableWidget cần cập nhật.
        row_label (str): Tên hàng (ví dụ: "15M").
        column_label (str): Tên cột (ví dụ: "Time").
        value (str): Giá trị cần điền vào ô.
    """
    # Tìm chỉ số hàng dựa trên tên hàng
    if Table == None:
        return
    row_index = None
    for row in range(Table.rowCount()):
        if Table.verticalHeaderItem(row) and Table.verticalHeaderItem(row).text() == row_label:
            row_index = row
            break

    # Tìm chỉ số cột dựa trên tên cột
    column_index = None
    for col in range(Table.columnCount()):
        if Table.horizontalHeaderItem(col) and Table.horizontalHeaderItem(col).text() == column_label:
            column_index = col
            break

    # Nếu tìm thấy cả hàng và cột, cập nhật giá trị
    if row_index is not None and column_index is not None:
        item = QTableWidgetItem(str(value))
        Table.setItem(row_index, column_index, item)
    else:
        print(f"Không tìm thấy hàng '{row_label}' hoặc cột '{column_label}'.")