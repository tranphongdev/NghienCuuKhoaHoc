import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
from keras.models import load_model

#Tải model đã train
model = load_model('model.h5')

#Các nhãn
classes = {
    1: 'Giới hạn tốc độ (20km/h)',
    2: 'Giới hạn tốc độ (30km/h)',
    3: 'Giới hạn tốc độ (50km/h)' ,
    4: 'Giới hạn tốc độ (60km/h)',
    5: 'Giới hạn tốc độ (70km/h)',
    6: 'Giới hạn tốc độ (80km/h)',
    7: 'Kết thúc giới hạn tốc độ (80km/h)',
    8: 'Giới hạn tốc độ (100km/h)',
    9: 'Giới hạn tốc độ (120km/h)',
    10: 'Cấm vượt',
    11: 'Cấm vượt đối với các phương tiện trên 3,5 tấn',
    12: 'Quyền ưu tiên ở giao lộ tiếp theo',
    13: 'Đường ưu tiên',
    14: 'Nhường đường',
    15: 'Dừng Lại',
    16: 'Cấm xe cơ giới',
    17: 'Cấm phương tiện trên 3,5 tấn',
    18: 'Cấm vào',
    19: 'Cảnh báo chung',
    20: 'Cung đường nguy hiểm bên trái',
    21: 'Cung đường nguy hiểm bên phải',
    22: 'Đường vòng hai chiều',
    23: 'Đường gập ghềnh',
    24: 'Đường trơn trượt',
    25: 'Đường hẹp bên phải',
    26: 'Công trường',
    27: 'Tín hiệu giao thông',
    28: 'Người đi bộ',
    29: 'Đường dành cho trẻ em',
    30: 'Đường dành cho xe đạp',
    31: 'Cảnh báo băng tuyết, đường trơn',
    32: 'Có động vật hoang dã băng qua',
    33: 'Kết thúc giới hạn tốc độ và vượt',
    34: 'Chỉ đường rẽ phải phía trước',
    35: 'Chỉ đường rẽ trái phía trước',
    36: 'Chỉ đường đi thẳng',
    37: 'Chỉ đường đi thẳng hoặc rẽ phải',
    38: 'Chỉ đường đi thẳng hoặc rẽ trái',
    39: 'Đi bên phải',
    40: 'Đi bên trái',
    41: 'Đi vòng qua',
    42: 'Kết thúc cấm vượt',
    43: 'Kết thúc cấm vượt xe trên 3.5 tấn'
}

#Tạo giao diện

# Khởi tạo cửa sổ
window = tk.Tk()
window.geometry('1500x900')
window.title("Nhận Diện Biển Báo Giao Thông")
window['background'] = '#CDCDCD'

label = Label(window, background='#CDCDCD', font=('Verdana', 25, 'bold'))
sign_image = Label(window)

#Hàm xử lý ảnh và dự đoán kết quả
def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((30, 30))
    image = np.expand_dims(image, axis=0) 
    image = np.array(image)
    print(image.shape)
    pred = model.predict(image)[0]
    sign = classes[np.argmax(pred) + 1]
    print(sign)
    label.configure(foreground='#FF0000', text= "BIỂN BÁO : " + sign)

#Tạo button dự đoán
def show_classify_button(file_path):
    classify_b = Button(window, text="Nhận diện", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('Verdana', 15, 'bold'))
    classify_b.place(relx=0.79, rely=0.50)

#Hàm up file ảnh
def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((window.winfo_width() / 2.25), (window.winfo_height() / 2.25))) #thu nhỏ hình ảnh
        im = ImageTk.PhotoImage(uploaded)

        # hình ảnh được hiển thị trong một nhãn sign_image, 
        # nhãn này được cập nhật thông qua phương thức configure() với tham số image=im
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except: #nếu như quá trình trên k thành công thì k hành động nào đc thực hiện bằng lênh pass
        pass
    
upload = Button(window, text="Tải ảnh lên", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('Verdana', 15, 'bold'))

upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(window, text="NHẬN DIỆN BIỂN BÁO GIAO THÔNG", pady=20, font=('Tahoma', 30, 'bold'))
heading.configure(background='#CDCDCD', foreground='#0000FF')
heading.pack()
heading = Label(window, text="CNĐT: TRẦN ĐÌNH PHONG", pady=20, font=('Tahoma', 24, 'bold'))
heading.configure(background='#CDCDCD', foreground='#0000FF')
heading.pack()
# Hiển thị cửa sổ
window.mainloop()