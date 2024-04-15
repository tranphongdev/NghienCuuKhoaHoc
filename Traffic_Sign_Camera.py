import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

#Viết văn bản trên ảnh 
threshold = 0.75  
font = cv2.FONT_HERSHEY_SIMPLEX

#Tải dữ liệu đã train
model = load_model('models.h5')

#Chuyển đổi thành ảnh xámqqq
def processing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

#các nhãn
def getCalssName(lenClasses):
    if lenClasses == 0:
        return 'Gioi hạn toc đo 20 km/h'
    elif lenClasses == 1:
        return 'Gioi han toc do 30 km/h'
    elif lenClasses == 2:
        return 'Gioi han toc do 50 km/h'
    elif lenClasses == 3:
        return 'Gioi han toc do 60 km/h'
    elif lenClasses == 4:
        return 'Gioi han toc do 70 km/h'
    elif lenClasses == 5:
        return 'Gioi han toc do 80 km/h'
    elif lenClasses == 6:
        return 'Toc do toi da 80km/h'
    elif lenClasses == 7:
        return 'Gioi han toc do 100 km/h'
    elif lenClasses == 8:
        return 'Gioi han toc do 120 km/h'
    elif lenClasses == 9:
        return 'Cam vuot'
    elif lenClasses == 10:
        return 'Cam phuong tien tren 3.5 tan'
    elif lenClasses == 11:
        return 'Bao hieu giao nhau voi duong khong uu tien'
    elif lenClasses == 12:
        return 'Duong uu tien'
    elif lenClasses == 13:
        return 'Nhuong quyen uu tien'
    elif lenClasses == 14:
        return 'Dung lai'
    elif lenClasses == 15:
        return 'Cam phuong tien'
    elif lenClasses == 16:
        return 'Cam xe tren 3.5 tan'
    elif lenClasses == 17:
        return 'Cam vao'
    elif lenClasses == 18:
        return 'Nguy hiem khac'
    elif lenClasses == 19:
        return 'Khuc cua nguy hiem ben trai'
    elif lenClasses == 20:
        return 'Khuc cua nguy hiem ben phai'
    elif lenClasses == 21:
        return 'Duong cong doi'
    elif lenClasses == 22:
        return 'Duong gap ghenh'
    elif lenClasses == 23:
        return 'Duong tron truot'
    elif lenClasses == 24:
        return 'Duong hep ben phai'
    elif lenClasses == 25:
        return 'Lam duong'
    elif lenClasses == 26:
        return 'Tin hieu giao thong'
    elif lenClasses == 27:
        return 'Duong di bo'
    elif lenClasses == 28:
        return 'Duong cho tre em'
    elif lenClasses == 29:
        return 'Duong cho xe dap'
    elif lenClasses == 30:
        return 'Coi chung bang tuyet'
    elif lenClasses == 31:
        return 'Duong cho dong vat hoang da'
    elif lenClasses == 32:
        return 'Khong gioi han toc do'
    elif lenClasses == 33:
        return 'Re phai ve phia truoc'
    elif lenClasses == 34:
        return 'Re trai ve phia truoc'
    elif lenClasses == 35:
        return 'Huong di thang phai theo'
    elif lenClasses == 36:
        return 'Duoc di thang va re phai'
    elif lenClasses == 37:
        return 'Duoc di thang va re trai'
    elif lenClasses == 38:
        return 'Di ben phai'
    elif lenClasses == 39:
        return 'Di ben trai'
    elif lenClasses == 40:
        return 'Noi giao nhau chay theo vong xoay'
    elif lenClasses == 41:
        return 'Ket thuc khong di qua'
    elif lenClasses == 42:
        return 'Het duong cam xe tren 3.5 tan'

#Khởi tạo camera
cap = cv2.VideoCapture(0)
while True:
    #Đọc ảnh từ camera
    ret, frame = cap.read()
    #Xử lý ảnh
    img = np.asarray(frame)
    img = cv2.resize(img, (32, 32))
    img = processing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(frame, "BIEN: ", (20, 35), font,0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "XAC XUAT: ", (20, 75),font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    
    #Dự đoán kết quả
    predictions = model.predict(img)
    classIndex = np.argmax(predictions, axis=-1)
    probabilityValue = np.amax(predictions)
    
    if probabilityValue > threshold:
        print(getCalssName(classIndex))
        cv2.putText(frame, str(classIndex) + " " + str(getCalssName(classIndex)), (120, 35),
                    font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, str(round(probabilityValue * 100, 2)) + " %", (180, 75),
                    font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow("Nhan Dien Bien Bao",frame)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()



