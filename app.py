import streamlit as st
import torch
import pickle
from vqa_model import VQAModel
import urllib.request
from PIL import Image

# =============================
# 1. Load Models & Encoders
# =============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.write(f"🚀 Using device: **{DEVICE}**")

MODEL_NAME = "ViT-L/14@336px"
NUM_CLASSES = 5239
MODEL_PATH = "Outputepoch_45.pth"  # ✅ Nên để file trong thư mục Output

# ✅ Load One-Hot Encoders (có thư mục Output trước tên file)
with open('Outputanswer_onehotencoder.pkl', 'rb') as f:
    ANSWER_ONEHOTENCODER = pickle.load(f)

with open('Outputanswer_type_onehotencoder.pkl', 'rb') as f:
    ANSWER_TYPE_ONEHOTENCODER = pickle.load(f)

# ✅ Load Model
model = VQAModel(num_classes=NUM_CLASSES, device=DEVICE,
                 hidden_size=512, model_name=MODEL_NAME).to(DEVICE)

# Nếu GPU có sẵn -> load model lên GPU
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

# =============================
# 2. Streamlit UI
# =============================
st.title("🖼️ Visual Question Answering (VQA Demo)")

image_url = st.text_input("🔗 Nhập link ảnh (hoặc bỏ trống để upload):")
uploaded_file = st.file_uploader("📁 Upload ảnh", type=['jpg', 'jpeg', 'png'])
question = st.text_input("❓ Nhập câu hỏi về bức ảnh:")

# =============================
# 3. Handle Image Input
# =============================
def save_image(image_url, uploaded_file):
    image_path = "user_image.jpg"
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        img.save(image_path)
        return image_path
    elif image_url:
        urllib.request.urlretrieve(image_url, image_path)
        return image_path
    return None

# =============================
# 4. Prediction
# =============================
if st.button("📌 Dự đoán"):
    image_path = save_image(image_url, uploaded_file)

    if not image_path or not question:
        st.warning("⚠️ Hãy chọn ảnh và nhập câu hỏi!")
    else:
        st.image(image_path, caption="Ảnh đã nhập", use_container_width=True)
        with st.spinner("⏳ Model đang xử lý..."):
            # model.test_model tự xử lý ảnh + văn bản
            pred_answer, pred_type, answerability = model.test_model(
                image_path=image_path,
                question=question
            )

            answer = ANSWER_ONEHOTENCODER.inverse_transform(
                pred_answer.cpu().detach().numpy())[0][0]
            type_ans = ANSWER_TYPE_ONEHOTENCODER.inverse_transform(
                pred_type.cpu().detach().numpy())[0][0]

        st.success("✅ **Kết quả dự đoán:**")
        st.write(f"- 📝 **Answer:** {answer}")
        st.write(f"- 📂 **Answer Type:** {type_ans}")
        st.write(f"- 🎯 **Answerability Score:** {answerability.item():.4f}")
