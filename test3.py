import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from ultralytics import YOLO

# ---------------------------------------------------------
# LUỒNG XỬ LÝ NỀN (Đọc Camera & Chạy YOLO)
# ---------------------------------------------------------
class YoloVideoThread(QThread):
    # Tín hiệu để gửi hình ảnh đã xử lý về luồng giao diện chính
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        print("[INFO] Đang tải mô hình YOLO...")
        model = YOLO('yolo11n.pt') # Dùng bản nano cho nhẹ
        
        print("[INFO] Khởi động Camera...")
        cap = cv2.VideoCapture(0) # Đổi thành địa chỉ RTSP nếu dùng camera IP

        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                # 1. Chạy suy luận YOLO
                results = model(frame, verbose=False)
                
                # 2. Lấy hình ảnh đã được vẽ Bounding Box
                annotated_frame = results[0].plot()

                # 3. Chuyển đổi định dạng màu (OpenCV dùng BGR, PyQt dùng RGB)
                rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                
                # 4. Ép kiểu sang QImage để PyQt có thể hiển thị
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                
                # Cắt/scale ảnh cho vừa với khung hiển thị
                p = convert_to_Qt_format.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                
                # Phóng tín hiệu mang hình ảnh gửi đi
                self.change_pixmap_signal.emit(p)
                
        # Dọn dẹp khi dừng luồng
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

# ---------------------------------------------------------
# LUỒNG GIAO DIỆN CHÍNH (Hiển thị & Nút bấm)
# ---------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hệ thống Giám sát Băng chuyền - YOLO11")
        self.resize(700, 600)

        # 1. Thiết lập các thành phần giao diện (Widgets)
        self.video_label = QLabel("Đang chờ tín hiệu camera...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white; font-size: 16px;")
        self.video_label.setFixedSize(640, 480)

        self.start_btn = QPushButton("Khởi động Hệ thống")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;")

        # 2. Sắp xếp bố cục (Layout)
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.start_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # 3. Khởi tạo Luồng chạy YOLO
        self.thread = YoloVideoThread()
        # Kết nối tín hiệu mang hình ảnh từ Thread tới hàm cập nhật UI
        self.thread.change_pixmap_signal.connect(self.update_image)
        
        # Kết nối nút bấm
        self.start_btn.clicked.connect(self.toggle_system)
        self.is_running = False

    def update_image(self, qt_image):
        # Hiển thị QImage lên QLabel
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def toggle_system(self):
        if not self.is_running:
            self.thread._run_flag = True
            self.thread.start()
            self.start_btn.setText("Dừng Hệ thống")
            self.start_btn.setStyleSheet("background-color: #dc3545; color: white; font-weight: bold;")
            self.is_running = True
        else:
            self.thread.stop()
            self.video_label.clear()
            self.video_label.setText("Hệ thống đã dừng")
            self.start_btn.setText("Khởi động Hệ thống")
            self.start_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;")
            self.is_running = False

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

# ---------------------------------------------------------
# KHỞI CHẠY ỨNG DỤNG
# ---------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())