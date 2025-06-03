import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import tensorflow.keras.layers as layers
import tensorflow as tf
import time
# --- Custom Layer untuk model ---
class SiluLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        return tf.nn.swish(inputs)
    def get_config(self):
        config = super().get_config()
        return config

# Load model sekali saat aplikasi start
@st.cache_resource
def load_my_model():
    model_path = 'D:\\Semester 6\\PDM\\model_ujicoba.keras'  # Ganti path sesuai kamu
    model = load_model(model_path, custom_objects={'Silu': SiluLayer})
    return model

model = load_my_model()

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Fungsi untuk convert landmarks jadi input image 256x256 grayscale
def landmarks_to_input_image(landmarks_list, img_size=256, point_radius=5):
    zero_hand = np.zeros(42, dtype=np.float32)
    hand1 = zero_hand
    hand2 = zero_hand

    if len(landmarks_list) == 1:
        hand1 = landmarks_list[0]
    elif len(landmarks_list) >= 2:
        avg_x_0 = np.mean(landmarks_list[0][0::2])
        avg_x_1 = np.mean(landmarks_list[1][0::2])
        if avg_x_0 < avg_x_1:
            hand_left = landmarks_list[0]
            hand_right = landmarks_list[1]
        else:
            hand_left = landmarks_list[1]
            hand_right = landmarks_list[0]
        hand1 = hand_right
        hand2 = hand_left

    x_coords = []
    y_coords = []
    for hand in [hand1, hand2]:
        x_coords.extend(hand[0::2])
        y_coords.extend(hand[1::2])

    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    x_coords = np.nan_to_num(x_coords, 0)
    y_coords = np.nan_to_num(y_coords, 0)

    valid_points_mask = (x_coords != 0) | (y_coords != 0)
    valid_x = x_coords[valid_points_mask]
    valid_y = y_coords[valid_points_mask]

    if len(valid_x) > 0 and len(valid_y) > 0:
        min_x, max_x = np.min(valid_x), np.max(valid_x)
        min_y, max_y = np.min(valid_y), np.max(valid_y)
    else:
        min_x, max_x = 0, 1
        min_y, max_y = 0, 1

    x_range = max_x - min_x if max_x - min_x > 0 else 1e-6
    y_range = max_y - min_y if max_y - min_y > 0 else 1e-6
    padding = 0.1
    effective_img_size = img_size * (1 - 2 * padding)
    scale_x = effective_img_size / x_range
    scale_y = effective_img_size / y_range
    offset_x = (img_size - (max_x - min_x) * scale_x) / 2
    offset_y = (img_size - (max_y - min_y) * scale_y) / 2

    canvas = np.zeros((img_size, img_size), dtype=np.uint8)
    for i in range(len(x_coords)):
        x, y = x_coords[i], y_coords[i]
        if valid_points_mask[i]:
            x_c = int((x - min_x) * scale_x + offset_x)
            y_c = int((y - min_y) * scale_y + offset_y)
            if 0 <= x_c < img_size and 0 <= y_c < img_size:
                cv2.circle(canvas, (x_c, y_c), point_radius, (255,), thickness=-1)

    canvas_norm = canvas.astype(np.float32) / 255.0
    canvas_norm = np.stack([canvas_norm]*3, axis=-1)  # 3 channel buat model
    return canvas_norm

# Kelas transformer video untuk streamlit-webrtc
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.labels = [chr(i) for i in range(ord('a'), ord('z')+1)]
        self.detected_chars = []

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.hands.process(img_rgb)
        landmarks_list = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks_2d = []
                for lm in hand_landmarks.landmark:
                    landmarks_2d.append(lm.x)
                    landmarks_2d.append(lm.y)
                landmarks_list.append(np.array(landmarks_2d, dtype=np.float32))

            # Prediksi dengan model
            try:
                input_img = landmarks_to_input_image(landmarks_list)
                input_tensor = np.expand_dims(input_img, axis=0)
                preds = model.predict(input_tensor, verbose=0)
                predicted_class_index = np.argmax(preds, axis=1)[0]
                confidence = preds[0][predicted_class_index]

                if 0 <= predicted_class_index < len(self.labels):
                    pred_char = self.labels[predicted_class_index].upper()
                else:
                    pred_char = "?"

                # Tampilkan prediksi jika confidence > 0.8
                if confidence > 0.8:
                    if not self.detected_chars or (self.detected_chars[-1] != pred_char):
                        self.detected_chars.append(pred_char)

                # Tampilkan teks di frame
                cv2.putText(img, f"Prediksi: {pred_char} ({confidence:.2f})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            except Exception as e:
                cv2.putText(img, f"Error: {str(e)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        return img

# Streamlit UI
st.title("Bahasa Isyarat SIBI dengan Webcam Streaming")

st.write("Deteksi huruf isyarat SIBI realtime menggunakan webcam dan model ML.")

webrtc_ctx = webrtc_streamer(
    key="sibi-sign",
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Tampilkan hasil karakter terdeteksi secara realtime (jika ada)
if webrtc_ctx.video_transformer:
    detected_chars = webrtc_ctx.video_transformer.detected_chars
    st.subheader("Karakter Terdeteksi:")
    placeholder = st.empty()
    st.write("".join(detected_chars))
    while True:
        detected_chars = webrtc_ctx.video_transformer.detected_chars
        placeholder.text("".join(detected_chars))
        time.sleep(0.5)  # update setiap 0.5 detik