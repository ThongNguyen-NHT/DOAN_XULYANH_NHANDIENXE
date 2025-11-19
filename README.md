# 🚗 Vehicle Counting System (Hệ thống Đếm Xe)

Hệ thống tự động phát hiện và đếm phương tiện giao thông (Xe máy, Ô tô, Xe buýt/Xe tải) từ video sử dụng **YOLOv4-tiny** để nhận diện và **Centroid Tracking** để theo dõi đối tượng.

![Demo Output](https://img.shields.io/badge/Demo-Running-green) 

## 📋 Tính năng chính
- **Phát hiện vật thể:** Sử dụng mô hình YOLOv4-tiny (nhẹ, nhanh) để nhận diện các loại xe.
- **Phân loại:** Tự động phân loại xe thành 3 nhóm: `Motorbike`, `Car`, `Bus/Truck`.
- **Theo dõi (Tracking):** Sử dụng thuật toán Centroid Tracking để gán ID cho từng xe, tránh đếm trùng lặp.
- **Đếm xe:** Đếm số lượng xe khi chúng đi qua vạch kẻ quy định (Counting Line).
- **Hiển thị:** Vẽ bounding box, ID xe và bảng thống kê số lượng trực tiếp trên video.

## 🛠 Yêu cầu hệ thống (Prerequisites)

Đảm bảo máy tính của bạn đã cài đặt **Python 3.7+**.
Các thư viện cần thiết:

- `opencv-python` (Xử lý ảnh & Computer Vision)
- `numpy` (Tính toán ma trận)
- `scipy` (Dùng tính khoảng cách Euclid trong tracker)

Cài đặt dependencies bằng lệnh:

```bash
pip install opencv-python numpy scipy