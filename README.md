# 🚘 License Plate Recognition System


## Tôi đang sử dụng GPU của Kaggle cho dự án này, và đây là liên kết đến notebook của tôi (nếu bạn không truy cập được, có thể do tôi đặt ở chế độ riêng tư):  
[Kaggle Notebook: license-plate](https://www.kaggle.com/code/nguyenquyetgiangson/license-plate)

## 📁 Cấu trúc thư mục

```

📂 license-plate/
├── 📁 imgs/
│   ├── 🖼️ sample1.jpg
│   └── 🖼️ sample2.jpg
├── ⚙️ .gitattributes
├── ⚙️ .gitignore
├── 🐍 app.py
├── 📄 best.onnx
├── 📄 license-plate.ipynb
└── 📄 requirements.txt

````

---

## 🚗 Giới thiệu dự án

**Mô tả ngắn gọn:**  
Dự án *License Plate Recognition* là một ứng dụng nhận diện biển số xe tự động, sử dụng **YOLO (ONNX)** để phát hiện vùng biển số và **EasyOCR** để đọc ký tự.  
Ứng dụng có giao diện thử nghiệm thân thiện bằng **Gradio**, cho phép người dùng upload ảnh và xem kết quả trực quan.

## 🎥 Demo hoạt động

Dưới đây là ví dụ minh họa cách ứng dụng nhận diện biển số hoạt động:

![Demo](https://raw.githubusercontent.com/GiangSon-5/license-plate/main/demo/demo.gif)



## 🌐 Triển khai trực tuyến

Bạn có thể trải nghiệm hệ thống nhận diện khuôn mặt qua giao diện web được triển khai tại Hugging Face Spaces:

👉 [License-plate trên Hugging Face](https://huggingface.co/spaces/GiangSon-5/license-plate)


### 🎯 Mục tiêu
- Phát hiện và khoanh vùng biển số xe trong ảnh.  
- Tiền xử lý vùng biển số nhằm cải thiện độ chính xác nhận diện ký tự.  
- Hiển thị kết quả nhanh qua giao diện web.

### ⚙️ Chức năng chính
- Tải mô hình YOLO (ONNX) để phát hiện vùng biển số.  
- Thực hiện các bước tiền xử lý ảnh (*deskew, threshold, morphology, sharpening*).  
- Nhận diện ký tự bằng **EasyOCR**, có áp dụng hậu xử lý (*rule-based corrections*).  
- Giao diện trực quan bằng **Gradio**: upload ảnh, xem gallery, kết quả text.

### 🧩 Công nghệ sử dụng
- **Python** (chính: `app.py`)  
- **Ultralytics YOLO (ONNX)**  
- **EasyOCR**, **OpenCV**, **Numpy**  
- **Gradio** (demo giao diện)  
- Các thư viện phụ trợ: chi tiết trong `requirements.txt`

---

## 🛠️ Hướng dẫn cài đặt

### 📋 Yêu cầu
- Python 3.9+ (khuyến nghị ≥ 3.10)  
- GPU có hỗ trợ CUDA (tùy chọn, nếu muốn tăng tốc)  
- Cài PyTorch tương thích thủ công nếu sử dụng GPU.  

### 🔧 Cài đặt môi trường

**1️⃣ Tạo và kích hoạt môi trường ảo**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
````

**2️⃣ Cài đặt thư viện**

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**3️⃣ (Tùy chọn) Cài PyTorch**

* CPU-only:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

* GPU (CUDA): xem hướng dẫn chính thức tại [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally)

💡 *Lưu ý:*

* Nên cài `torch` **trước** `ultralytics` để tránh lỗi phụ thuộc.
* Nếu ONNX load thất bại, kiểm tra lại `best.onnx` có bị hỏng không.

---

## ▶️ Cách chạy chương trình

1. Kích hoạt môi trường ảo (như trên).
2. Chạy ứng dụng:

```powershell
python app.py
```

📍 Khi chạy thành công, Gradio sẽ hiển thị URL (thường là `http://127.0.0.1:7860`).

### 🪶 Giao diện Gradio

* **Ảnh kết quả:** hiển thị bounding box và text.
* **Gallery processed:** ảnh vùng biển số sau tiền xử lý OCR.
* **Gallery cropped:** vùng crop từ YOLO.
* **Text output:** chuỗi biển số nhận diện (vd: `30A-123.45`).

🧪 *Ví dụ nhanh:*
Mở `imgs/sample1.jpg` hoặc `imgs/sample2.jpg`, sau đó bấm “Nhận diện 🔎”.

---

## 🔁 Luồng thực hiện (Workflow)

1. **Chuẩn bị dữ liệu:** thu thập và gắn nhãn ảnh xe → lưu trong `imgs/`.
2. **Huấn luyện YOLO:** dùng `ultralytics` hoặc notebook để tạo `best.onnx`.
3. **Triển khai:** đặt `best.onnx` trong thư mục gốc → chạy `app.py`.
4. **Đánh giá:** kiểm tra kết quả và tinh chỉnh bước tiền xử lý hoặc hậu xử lý.

📚 *Pipeline tổng quan:*
YOLO (detect) → crop → preprocess → EasyOCR → post-processing.

---

## 📸 Kết quả & Demo

Dự án cung cấp sẵn ảnh mẫu trong `imgs/`.
Khi chạy demo:

* Ảnh hiển thị bounding box.
* Gallery các vùng crop & tiền xử lý.
* Text kết quả tổng hợp.

💡 *Ví dụ kết quả:*

* Ảnh đầu vào: xe trên đường.
* Kết quả: `Biển số 1: 30A12345` *(tùy theo model & ánh sáng)*.

---

## 🧩 Lỗi phổ biến & Gợi ý khắc phục

| ⚠️ Vấn đề                       | 💬 Nguyên nhân & Giải pháp                        |
| ------------------------------- | ------------------------------------------------- |
| `Error loading YOLO ONNX model` | Kiểm tra file `best.onnx` có tồn tại hoặc bị hỏng |
| `EasyOCR load error`            | Cần internet để tải model lần đầu                 |
| `Gradio UI không khởi chạy`     | Kiểm tra biến `yolo_model` & `reader` trong code  |

---

## ✍️ Tác giả 

* **Tác giả:** GiangSon-5



