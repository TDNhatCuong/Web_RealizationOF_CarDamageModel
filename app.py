from flask import Flask, request, render_template, send_file
import torch
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from PIL import Image
import numpy as np
import cv2
import io
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# C:\Users\Acer\detectron2_env\Scripts\activate

# Cấu hình Detectron2 và tải mô hình
def load_model():
    cfg = get_cfg()
    cfg.merge_from_file("./model/config.yaml")      # Đường dẫn đến file cấu hình của Detectron2
    cfg.MODEL.WEIGHTS = "./model/model_final.pth"   # Đường dẫn tới model_final.pth

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    # Đăng ký metadata cho nhãn 'damage'
    MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes = ['damage']

    return DefaultPredictor(cfg), cfg  #


predictor, cfg = load_model()


@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        # Kiểm tra có file ảnh nào được tải lên không
        if 'file' not in request.files:
            return "No file uploaded", 400
        file = request.files['file']

        if file:
            # Lưu ảnh được tải lên
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Đọc file ảnh và chuyển sang định dạng numpy array
            img = Image.open(file_path)
            img_array = np.array(img)

            # Chuyển đổi ảnh RGB sang BGR vì OpenCV yêu cầu định dạng BGR
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # Thực hiện dự đoán với Detectron2
            outputs = predictor(img_bgr)

            # Vẽ dự đoán lên ảnh gốc với metadata từ 'damage'
            v = Visualizer(img_bgr[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

            # Chuyển kết quả về dạng PIL để gửi lại cho người dùng
            result_img = Image.fromarray(out.get_image()[:, :, ::-1])
            result_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + file.filename)
            result_img.save(result_img_path)

            # Gửi lại hình ảnh gốc và kết quả về giao diện
            return render_template('index.html', input_image=file_path, result_image=result_img_path)

    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
