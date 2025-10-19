import gradio as gr
import cv2
import numpy as np
import easyocr
import math
import re
import string
from ultralytics import YOLO
import gc 

# 1. Tải model EasyOCR
try:
    reader = easyocr.Reader(['en'])
    print("EasyOCR reader loaded successfully.")
except Exception as e:
    print(f"Error loading EasyOCR reader: {e}")
    reader = None

# 2. Tải model YOLOv từ file .onnx
try:
    yolo_model = YOLO("best.onnx")
    print("YOLO ONNX model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO ONNX model 'best.onnx': {e}")
    print("Please make sure 'best.onnx' is in the same directory.")
    yolo_model = None

# --- Các hàm xử lý ảnh  ---

def rotate_image(image, angle):
    """Xoay ảnh một góc (angle)"""
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def compute_skew(src_img):
    """Tính toán góc nghiêng của ảnh"""
    if len(src_img.shape) == 3:
        h, w, _ = src_img.shape
    elif len(src_img.shape) == 2:
        h, w = src_img.shape
    else:
        print('Unsupported image type')
        return 0.0

    img = cv2.medianBlur(src_img, 3)
    edges = cv2.Canny(img, threshold1=50, threshold2=170, apertureSize=3, L2gradient=True)
    
    if h > 55:
        edges = edges[min(55, h-1):,:]

    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, 30, minLineLength=w / 4.0, maxLineGap=h / 4.0)
    
    if lines is None:
        return 0.0

    angle = 0.0
    nlines = lines.shape[0]
    
    cnt = 0
    for x1, y1, x2, y2 in lines[:, 0]:
        ang = np.arctan2(y2 - y1, x2 - x1)
        if math.fabs(ang) <= 30 / 180 * math.pi:
            angle += ang
            cnt += 1

    if cnt == 0:
        return 0.0
    return (angle / cnt) * 180 / math.pi

def deskew(src_img):
    """Làm thẳng ảnh bị nghiêng"""
    return rotate_image(src_img, compute_skew(src_img))

def preprocess_v3(oimg):
    """Hàm tiền xử lý ảnh  để cải thiện OCR"""
    try:
        dimg = deskew(oimg)
        img = cv2.cvtColor(dimg, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(img, 140, 255, cv2.THRESH_BINARY)
        img = cv2.GaussianBlur(img, (5, 5), 0)

        rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 5))
        blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, rectKern)

        squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        light = cv2.morphologyEx(img, cv2.MORPH_CLOSE, squareKern)
        light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        rows, cols = np.where(light == 255)
        if rows.size == 0 or cols.size == 0:
            return oimg

        top, bottom = rows.min(), rows.max()
        left, right = cols.min(), cols.max()

        cropped = dimg[top:bottom, left:right]
        if cropped.size == 0:
            return oimg

        img = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        median_val = np.median(img.reshape(-1,), axis=0)
        mean_val = np.mean(img.reshape(-1,), axis=0)
        _, img = cv2.threshold(img, (median_val + mean_val) / 2, 255, cv2.THRESH_BINARY)
        img = cv2.GaussianBlur(img, (7, 7), 0)

        recolor = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        sharpened_image2 = cv2.addWeighted(cropped, 0.5, recolor, 0.5, 0)
        for i in range(3, 10, 2):
            blurred = cv2.GaussianBlur(sharpened_image2, (i, i), 1)
            sharpened_image2 = cv2.addWeighted(sharpened_image2, 1.5, blurred, -0.5, 0)
        
        return sharpened_image2
    except Exception as e:
        print(f"Error in preprocess_v3: {e}")
        return oimg


dict_char_to_int = {'O': '0',
                    'L': '4',
                    'B': '8',
                    'I': '1',
                    'T': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'Z': '2',
                    'S': '5'}

dict_int_to_char = {'0': 'C',
                    '1': 'T',
                    '4': 'A',
                    '6': 'G',
                    '2': 'Z',
                    '5': 'S',
                    '8': 'B'}


def remove_symbols(result):
    """Xóa các ký tự đặc biệt"""
    return re.sub(r'[^A-Z\d]', '', result)

def post_process(list_results, letter_idx=2):
    """Hậu xử lý text nhận diện """
    predict_results = []
    for result in list_results:
        if len(result) < 2:
            predict_results.append(result)
            continue

        # Dòng trên
        if len(result[0]) > letter_idx:
            letter = result[0][letter_idx]
            if letter in dict_int_to_char.keys():
                result[0] = result[0][:letter_idx] + dict_int_to_char[letter] + result[0][letter_idx+1:]

        for i in range(3):
            if i != letter_idx:
                if i < len(result[0]):
                    number = result[0][i]
                    if number in dict_char_to_int.keys():
                        result[0] = result[0][:i] + dict_char_to_int[number] + result[0][i+1:]

        if len(result[0]) >= 4:
            char_var = result[0][3]
            if char_var not in ["A", "B"]:
                if char_var in dict_char_to_int.keys():
                    result[0] = result[0][:3] + dict_char_to_int[char_var]

        # Dòng dưới
        for i in range(len(result[1])):
            number = result[1][i]
            if number in dict_char_to_int.keys():
                result[1] = result[1].replace(number, dict_char_to_int[number])
        
        predict_results.append(result)
    return predict_results

# ---  Hàm xử lý chính cho Gradio ---

def recognize_license_plate(input_image):
    """
    Hàm chính thực hiện toàn bộ pipeline:
    1. Phát hiện TẤT CẢ biển số bằng YOLO
    2. Lặp qua từng biển số:
        a. Cắt vùng biển số (crop)
        b. Xử lý ảnh (deskew, preprocess_v3)
        c. Nhận diện chữ bằng EasyOCR
        d. Hậu xử lý text
    3. Vẽ bounding box và text lên ảnh kết quả
    4. Trả về: Ảnh kết quả, danh sách ảnh đã xử lý, danh sách ảnh đã cắt, chuỗi văn bản tổng hợp
    """
    if yolo_model is None or reader is None:
        # Trả về 4 giá trị rỗng/mặc định để khớp với outputs của Gradio
        return input_image, None, None, "Lỗi: Model chưa được tải."

    # Tạo bản sao để vẽ lên
    image_with_boxes = input_image.copy()
    
    # 1. Phát hiện biển số bằng YOLO
    results = yolo_model.predict(
        input_image,
        imgsz=480,
        conf=0.5, 
        verbose=False
    )

    if not results or not results[0].boxes:
        # Giải phóng bộ nhớ ngay cả khi không tìm thấy gì
        gc.collect() 
        return input_image, None, None, "Không phát hiện thấy biển số."

    # Khởi tạo các danh sách để lưu kết quả
    all_cropped_plates = []
    all_processed_plates = []
    all_recognized_texts = []

    # 2. Lặp qua TẤT CẢ các biển số được phát hiện
    for box in results[0].boxes:
        try:
            # 2a. Cắt vùng biển số
            coords = box.xyxy[0].cpu().numpy().astype(int)
            [x1, y1, x2, y2] = coords
            x1, y1, x2, y2 = max(0, x1), max(0, y1), max(0, x2), max(0, y2)
            
            cropped_plate = input_image[y1:y2, x1:x2]
            if cropped_plate.size == 0:
                continue # Bỏ qua nếu ảnh cắt bị rỗng
            
            all_cropped_plates.append(cropped_plate)

            # 2b. Xử lý ảnh
            processed_plate = preprocess_v3(cropped_plate)
            all_processed_plates.append(processed_plate)

            # 2c. Nhận diện chữ
            allow_list = string.ascii_uppercase + string.digits + '-.'
            ocr_result_raw = reader.readtext(processed_plate, detail=0, allowlist=allow_list)

            recognized_text = "N/A" # Giá trị mặc định nếu OCR thất bại
            if ocr_result_raw:
                # 2d. Hậu xử lý text
                cleaned_results = [remove_symbols(r.upper()) for r in ocr_result_raw]
                cleaned_results = [r for r in cleaned_results if r]

                if cleaned_results:
                    if len(cleaned_results) >= 2:
                        try:
                            lines_to_process = [cleaned_results[0], cleaned_results[1]]
                            final_result_list = post_process([lines_to_process])
                            recognized_text = "".join(final_result_list[0])
                        except Exception as e:
                            print(f"Post-processing failed: {e}. Falling back.")
                            recognized_text = cleaned_results[0] + cleaned_results[1]
                    else:
                        recognized_text = "".join(cleaned_results)

            all_recognized_texts.append(recognized_text)
            
            # 3. Vẽ bounding box và text lên ảnh kết quả
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_with_boxes, recognized_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        except Exception as e:
            print(f"Error processing box: {e}")
            continue # Bỏ qua box này nếu có lỗi

    # 4. Chuẩn bị chuỗi văn bản tổng hợp để trả về
    if not all_recognized_texts:
        # Trường hợp YOLO phát hiện nhưng tất cả các bước sau đều lỗi
        gc.collect() 
        return input_image, None, None, "Phát hiện được biển số nhưng không đọc được chữ."

    final_text_output = ""
    for i, txt in enumerate(all_recognized_texts):
        final_text_output += f"Biển số {i+1}: {txt}\n"

    # Chạy bộ dọn rác thủ công trước khi trả về kết quả
    gc.collect()

    # Trả về 4 giá trị
    return image_with_boxes, all_processed_plates, all_cropped_plates, final_text_output

# ---  Xây dựng giao diện Gradio ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Ứng dụng nhận diện biển số xe 🚗
        Upload ảnh xe của bạn để hệ thống phát hiện và nhận diện biển số.
        """
    )

    with gr.Row():
        image_input = gr.Image(type="numpy", label="Upload ảnh tại đây")

    # Thêm ảnh mẫu từ thư mục imgs
    gr.Examples(
        examples=[
            "imgs/sample1.jpg",
            "imgs/sample2.jpg"
        ],
        inputs=image_input
    )

    recognize_button = gr.Button("Nhận diện 🔎", variant="primary")

    with gr.Accordion("Kết quả chi tiết", open=True):
        # Output 1: Ảnh kết quả (vẽ tất cả box)
        result_image_output = gr.Image(label="Ảnh kết quả (Tất cả biển số)")

        # Output 4: Kết quả nhận diện (dạng văn bản)
        text_output = gr.Textbox(label="Kết quả nhận diện (EasyOCR)", lines=5)

        with gr.Row():
            # Output 3: Gallery các ảnh đã cắt (từ YOLO)
            cropped_gallery = gr.Gallery(label="Các vùng biển số (YOLO crop)", columns=4, height=200)

            # Output 2: Gallery các ảnh đã xử lý (cho OCR)
            processed_gallery = gr.Gallery(label="Biển số đã xử lý (cho OCR)", columns=4, height=200)

    # Liên kết nút bấm với hàm xử lý
    recognize_button.click(
        fn=recognize_license_plate,
        inputs=image_input,
        outputs=[result_image_output, processed_gallery, cropped_gallery, text_output]
    )

    gr.Markdown("--- \n *Powered by Gradio, YOLOv, and EasyOCR*")

# Chạy ứng dụng
if __name__ == "__main__":
    if yolo_model is None or reader is None:
        print("Models failed to load. Gradio app will not start.")
    else:
        print("Starting Gradio app...")
        demo.launch(debug=True)