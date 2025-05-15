import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QSize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class HandwritingApp(QMainWindow):
    def __init__(self, model_path):
        super().__init__()
        self.model = self.load_model(model_path)
        self.initUI()
        
    def load_model(self, path):
        # 加载训练好的模型
        model = CNN()
        model.load_state_dict(torch.load(path))
        model.eval()
        return model
        
    def initUI(self):
        # 主窗口设置
        self.setWindowTitle("手写数字识别")
        self.setFixedSize(400, 500)
        
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 布局
        layout = QVBoxLayout()
        
        # 画布设置
        self.canvas = QLabel()
        self.canvas.setMinimumSize(280, 280)
        self.canvas.setStyleSheet("background-color: white; border: 2px solid #ccc;")
        self.image = QImage(QSize(280, 280), QImage.Format_RGB32)
        self.image.fill(Qt.white)
        
        # 结果显示标签
        self.result_label = QLabel("请在手写板上书写数字")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 24px; color: #333; margin: 20px 0;")
        
        # 清除按钮
        clear_btn = QPushButton("清除画布")
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 10px;
                font-size: 16px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        clear_btn.clicked.connect(self.clear_canvas)
        
        # 识别按钮
        recognize_btn = QPushButton("识别数字")
        recognize_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                font-size: 16px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #388E3C;
            }
        """)
        recognize_btn.clicked.connect(self.recognize_digit)
        
        # 添加部件到布局
        layout.addWidget(self.canvas, alignment=Qt.AlignCenter)
        layout.addWidget(self.result_label)
        layout.addWidget(recognize_btn)
        layout.addWidget(clear_btn)
        
        central_widget.setLayout(layout)
        
    def mouseMoveEvent(self, event):
        # 鼠标移动事件处理
        if event.buttons() == Qt.LeftButton:
            painter = QPainter(self.image)
            painter.setPen(QPen(Qt.black, 15, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawPoint(event.pos() - self.canvas.pos())
            painter.end()
            self.update_canvas()
    
    def update_canvas(self):
        # 更新画布显示
        self.canvas.setPixmap(QPixmap.fromImage(self.image))
        
    def clear_canvas(self):
        # 清除画布
        self.image.fill(Qt.white)
        self.update_canvas()
        self.result_label.setText("请在手写板上书写数字")
        
    def recognize_digit(self):
        # 识别手写数字
        img = self.image.scaled(28, 28).convertToFormat(QImage.Format_Grayscale8)
        ptr = img.bits()
        ptr.setsize(img.byteCount())
        arr = np.array(ptr).reshape(28, 28).astype(np.float32)
        
        # 增强预处理
        arr = 255 - arr  # 反色处理(手写板是白底黑字，MNIST是黑底白字)
        arr = np.clip(arr, 0, 255)  # 确保值在0-255范围内
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        tensor = transform(arr).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(tensor)
            pred = output.argmax(dim=1).item()
        
        self.result_label.setText(f"识别结果: {pred} (置信度: {F.softmax(output, dim=1)[0][pred].item():.2f})")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HandwritingApp("./final_model.pth")
    window.show()
    sys.exit(app.exec_())