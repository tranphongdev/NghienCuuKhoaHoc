import tensorflow as tf  #Đào tạo mô hình
import numpy as np #Xử lý mảng
import pandas as pd  #làm việc với dữ liệu dạng table
import matplotlib.pyplot as plt #vẽ đồ thị
from PIL import Image #thay đổi kích thước hình ảnh
import os #truy cập đến các thư mục ảnh
from sklearn.model_selection import train_test_split #hàm chia dữ liệu thành các tập train và test
from keras.utils import to_categorical #hàm mã hóa các nhãn của tập train và test
from keras.models import Sequential, load_model #xây dựng mô hình
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout #các lớp

#Khởi tạo các biến
data = [] 
labels = []
classes = 43 #số lương thư mục ảnh
cur_path = os.getcwd()

# Truy xuất hình ảnh và nhãn của chúng
for i in range(classes):
    path = os.path.join(cur_path, 'train', str(i))
    images = os.listdir(path)

    for a in images:
        try:
            image = Image.open(path + '\\' + a)
            image = image.resize((30, 30))
            image = np.array(image)

            data.append(image) #sau khi resize thêm vào cuối dsach
            labels.append(i)
        except:
            print("Load ảnh lỗi")
            
#Chuyển đổi danh sách thành mảng trống
data = np.array(data)
labels = np.array(labels)
print(data.shape, labels.shape)

# Tách tập dữ liệu huấn luyện và thử nghiệm
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42) #ts = 0.2 > 0.8 còn lại là tập train
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Chuyển đổi các nhãn thành một mã hóa dạng one-hot encoding [0, 0, 0, 0, 1, 0, ..., 0]
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

# Xây dựng mô hình
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25)) #tỉ lệ dropout là 0.25 để tránh overfitting(độ chính xác cao nhưng k hoạt động tổt trên tập dữ liệu test hoặc mới)
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax')) #Lớp fully connected cuối cùng với 43 thư mục và hàm kích hoạt là softmax để dự đoán xác suất của mỗi lớp

# Tổng hợp mô hình
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
#cata: hàm mất mát, adam thuật toán tối ưu, metric chọn độ đo đánh giá hiệu suất của mô hình 

#Train mô hình
epochs = 15
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test)) #bsize: mẫu dữ liệu, val đánh giá mô hình trong lúc train
model.save("model.h5") #lưu file train

# vẽ đồ thị về đánh giá độ chính sách sau mỗi lần epochs
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
# vẽ đồ thị về đánh giá độ mất mát
plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# Kiểm tra độ chính xác trên tập dữ liệu Test
from sklearn.metrics import accuracy_score

y_test = pd.read_csv('Test.csv')
labels = y_test["ClassId"].values
imgs = y_test["Path"].values
data = []

for img in imgs:
    image = Image.open(img)
    image = image.resize((30, 30))
    data.append(np.array(image))

X_test = np.array(data) #xtest lưu mảng np trong tập test
pred = np.argmax(model.predict(X_test), axis=-1) #biến pred để lưu kết quả dự đoán

#  Tính độ chính xác của mô hình trên tập dữ liệu test
from sklearn.metrics import accuracy_score
print(accuracy_score(labels, pred))