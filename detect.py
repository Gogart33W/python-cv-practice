import cv2
import numpy as np
import torch
from PIL import Image
import onnxruntime as ort
from ultralytics.utils.nms import non_max_suppression
from utils import load_toml_as_dict


class Detect:
    def __init__(self, model_path, ignore_classes=None, classes=None, input_size=(640, 640)):
        cfg = load_toml_as_dict("cfg/general_config.toml")
        self.preferred_device = cfg.get('cpu_or_gpu', 'auto')

        self.model_path = model_path
        self.classes = classes
        self.ignore_classes = ignore_classes if ignore_classes else []
        self.input_size = input_size

        self.model, self.device = self.load_model()

    def load_model(self):
        available_providers = ort.get_available_providers()

        # 🔥 SMART PROVIDER SELECTION (FIXED)
        onnx_provider = None

        if self.preferred_device in ["gpu", "auto"]:

            # 1. DirectML (best for Windows now)
            if "DmlExecutionProvider" in available_providers:
                onnx_provider = "DmlExecutionProvider"
                print("Using GPU (DirectML)")

            # 2. CUDA (only if fully working)
            elif "CUDAExecutionProvider" in available_providers:
                onnx_provider = "CUDAExecutionProvider"
                print("Using CUDA GPU")

            # 3. fallback GPU providers (optional)
            elif "AzureExecutionProvider" in available_providers:
                onnx_provider = "AzureExecutionProvider"
                print("Using Azure GPU provider")

            # 4. CPU fallback
            else:
                onnx_provider = "CPUExecutionProvider"
                print("Using CPU (no GPU provider found)")

        else:
            onnx_provider = "CPUExecutionProvider"
            print("Forced CPU mode")

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        model = ort.InferenceSession(
            self.model_path,
            sess_options=so,
            providers=[onnx_provider]
        )

        return model, onnx_provider

    def preprocess_image(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)

        h, w, _ = img.shape
        scale = min(self.input_size[0] / h, self.input_size[1] / w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        padded_img = np.full(
            (self.input_size[0], self.input_size[1], 3),
            128,
            dtype=np.uint8
        )
        padded_img[:new_h, :new_w, :] = resized_img

        padded_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        padded_img = padded_img.astype(np.float32) / 255.0
        padded_img = np.transpose(padded_img, (2, 0, 1))
        padded_img = np.expand_dims(padded_img, axis=0)

        return torch.from_numpy(padded_img), new_w, new_h

    def postprocess(self, preds, img, orig_img_shape, resized_shape, conf_tresh=0.6):

        preds = non_max_suppression(
            preds,
            conf_thres=conf_tresh,
            iou_thres=0.6,
            classes=None,
            agnostic=False,
        )

        orig_h, orig_w = orig_img_shape
        resized_w, resized_h = resized_shape

        scale_w = orig_w / resized_w
        scale_h = orig_h / resized_h

        results = []

        for pred in preds:
            if len(pred):
                pred[:, 0] *= scale_w
                pred[:, 1] *= scale_h
                pred[:, 2] *= scale_w
                pred[:, 3] *= scale_h
                results.append(pred.cpu().numpy())

        return results

    def detect_objects(self, img, conf_tresh=0.6):

        if isinstance(img, Image.Image):
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        orig_h, orig_w = img.shape[:2]
        orig_img_shape = (orig_h, orig_w)

        preprocessed_img, resized_w, resized_h = self.preprocess_image(img)
        resized_shape = (resized_w, resized_h)

        outputs = self.model.run(
            None,
            {'images': preprocessed_img.cpu().numpy()}
        )

        detections = self.postprocess(
            torch.from_numpy(outputs[0]),
            preprocessed_img,
            orig_img_shape,
            resized_shape,
            conf_tresh
        )

        results = {}

        for detection in detections:
            for *xyxy, conf, cls in detection:

                x1, y1, x2, y2 = map(int, xyxy)
                class_id = int(cls)
                class_name = self.classes[class_id]

                if class_id in self.ignore_classes or class_name in self.ignore_classes:
                    continue

                if class_name not in results:
                    results[class_name] = []

                results[class_name].append([x1, y1, x2, y2])

        return results