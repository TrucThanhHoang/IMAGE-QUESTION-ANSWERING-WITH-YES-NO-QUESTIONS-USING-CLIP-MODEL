import streamlit as st
import torch
import pickle
from vqa_model import VQAModel
import urllib.request
from PIL import Image

# =============================
# 1. Load Models & Encoders
# =============================
DEVICE = torch.device("cpu")
MODEL_NAME = "ViT-L/14@336px"
NUM_CLASSES = 5410
MODEL_PATH = "Saved_Models/model.pth"

# Load One-Hot Encoders
with open('Saved_Models/answer_onehotencoder.pkl', 'rb') as f:
    ANSWER_ONEHOTENCODER = pickle.load(f)
with open('Saved_Models/answer_type_onehotencoder.pkl', 'rb') as f:
    ANSWER_TYPE_ONEHOTENCODER = pickle.load(f)

# Load Model
model = VQAModel(num_classes=NUM_CLASSES, device=DEVICE,
                 hidden_size=512, model_name=MODEL_NAME).to(DEVICE)
model.load_model(MODEL_PATH)

# =============================
# 2. Streamlit UI
# =============================
st.title("🖼️ Visual Question Answering (MiniGPT-4 / CLIP)")

# Input: URL or File
image_url = st.text_input("🔗 Nhập đường link ảnh (hoặc bỏ trống nếu upload):")
uploaded_file = st.file_uploader("📁 Upload ảnh từ máy tính", type=['jpg', 'jpeg', 'png'])
question = st.text_input("❓ Nhập câu hỏi về bức ảnh:")

# =============================
# 3. Handle Image
# =============================
def save_image_from_input(image_url, uploaded_file):
    image_path = 'user_image.jpg'
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image.save(image_path)
        return image_path
    elif image_url:
        urllib.request.urlretrieve(image_url, image_path)
        return image_path
    else:
        return None

# =============================
# 4. Predict
# =============================
if st.button("📌 Dự đoán"):
    image_path = save_image_from_input(image_url, uploaded_file)

    if image_path is None or question.strip() == "":
        st.warning("⚠️ Hãy chọn ảnh và nhập câu hỏi!")
    else:
        st.image(image_path, caption="Ảnh bạn đã chọn", use_container_width=True)

        with st.spinner("⏳ Đang suy nghĩ..."):
            pred_answer, pred_answer_type, answerability = model.test_model(
                image_path=image_path,
                question=question
            )

            answer = ANSWER_ONEHOTENCODER.inverse_transform(
                pred_answer.cpu().detach().numpy())[0][0]
            answer_type = ANSWER_TYPE_ONEHOTENCODER.inverse_transform(
                pred_answer_type.cpu().detach().numpy())[0][0]

        st.success("✅ **Kết quả dự đoán:**")
        st.write(f"- 📝 **Answer:** {answer}")
        st.write(f"- 📂 **Answer Type:** {answer_type}")
        st.write(f"- 🎯 **Answerability score:** {answerability.item():.4f}")

