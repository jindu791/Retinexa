import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import streamlit as st

# Path to your trained model
MODEL_PATH = "models/cataract_resnet18_binary.pth"


# ===========================
# 1. Risk decision function
# ===========================
def cataract_risk_decision(p, age, sex, smoker, alcohol_level):
    """
    p: float, model probability of cataract from image [0,1]
    age: int
    sex: 'Male' or 'Female' or 'Other'
    smoker: 0 or 1
    alcohol_level: 0 (none), 1 (moderate), 2 (heavy)
    """

    # Very confident from image alone
    if p >= 0.8:
        return "High risk (refer)"
    if p <= 0.1:
        return "Very low risk"

    # Borderline zone – use risk factors
    score = 0

    # age contribution
    if age >= 70:
        score += 2
    elif age >= 50:
        score += 1

    # simple sex factor (slightly higher risk in older females)
    if sex == "Female" and age >= 60:
        score += 1

    # smoking
    if smoker == 1:
        score += 1

    # alcohol
    if alcohol_level == 2:
        score += 2
    elif alcohol_level == 1:
        score += 1

    # combine
    if p >= 0.5 and score >= 2:
        return "High risk (refer)"
    if p >= 0.5 and score < 2:
        return "Medium risk (monitor / follow-up)"
    if p < 0.5 and score >= 3:
        return "Medium risk (monitor / follow-up)"
    else:
        return "Low risk"


# ===========================
# 2. Load model + transforms
# ===========================
@st.cache_resource
def load_model_and_transform():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)  # cataract(0), normal(1)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            f"Train it first with train.py"
        )

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    img_size = 224
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return model, device, transform


# ===========================
# 3. Predict cataract prob
# ===========================
def predict_cataract_prob(model, device, transform, pil_image: Image.Image) -> float:
    img = pil_image.convert("RGB")
    x = transform(img).unsqueeze(0).to(device)  # [1,3,H,W]

    with torch.no_grad():
        logits = model(x)   # [1,2]

        # Temperature scaling to soften probabilities a bit
        T = 2.0
        probs = F.softmax(logits / T, dim=1)[0]

        # index 0 = cataract (from training: {'cataract': 0, 'normal': 1})
        p_cataract = probs[0].item()

    return p_cataract


# ===========================
# 4. Streamlit UI
# ===========================
def main():
    st.set_page_config(page_title="Cataract Screening Demo", page_icon="👁️", layout="wide")

    # =========================
    # YOUR LOGO AT THE TOP
    # =========================
    st.image("retinexa.png", width=500)  # <-- adjust width if needed

    st.title("AI Screening Assistant Demo: Cataract Detection from Retina Images")


    col_left, col_right = st.columns([1, 1.2])

    # ---------------- LEFT COLUMN: INPUTS ----------------
    with col_left:
        st.subheader("Step 1 – Upload retina image")
        uploaded_file = st.file_uploader(
            "Upload a fundus (retina) image", type=["png", "jpg", "jpeg"]
        )

        st.subheader("Step 2 – Patient details")
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=60, step=1)

        sex = st.selectbox("Sex", ["Male", "Female", "Other"], index=0)

        smoker_str = st.selectbox("Do you smoke?", ["No", "Yes"], index=0)
        smoker = 1 if smoker_str == "Yes" else 0

        alcohol_str = st.selectbox(
            "Alcohol consumption",
            ["None", "Moderate", "Heavy"],
            index=0,
        )
        alcohol_level = {"None": 0, "Moderate": 1, "Heavy": 2}[alcohol_str]

        run_button = st.button("Run assessment")

    # ---------------- RIGHT COLUMN: OUTPUT ----------------
    with col_right:
        st.subheader("Retina image & result")

        pil_img = None
        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file)
            st.image(pil_img, caption="Uploaded retina image", use_column_width=True)
        else:
            st.info("Upload an image on the left to see it here.")

        if run_button:
            if pil_img is None:
                st.error("Please upload an image first.")
            else:
                with st.spinner("Analysing image and risk factors..."):
                    model, device, transform = load_model_and_transform()
                    p = predict_cataract_prob(model, device, transform, pil_img)

                    # clamp probability into a safe range
                    p_clamped = min(max(p, 0.05), 0.95)

                    # overall risk category (text)
                    risk = cataract_risk_decision(p_clamped, age, sex, smoker, alcohol_level)

                    # compute both probabilities
                    p_cataract = p_clamped
                    p_normal = 1 - p_clamped

                    # choose which confidence to show based on which is larger
                    if p_cataract > p_normal:
                        display_label = "Chance of cataract"
                        display_value = p_cataract * 100
                        main_emoji = "🔴"
                    else:
                        display_label = "Chance of NO cataract"
                        display_value = p_normal * 100
                        main_emoji = "🟢"

                st.markdown("---")
                st.subheader("Assessment result")

                # big confidence metric
                st.metric(
                    label=f"{main_emoji} {display_label}",
                    value=f"{display_value:.1f} %",
                )

                # show both probabilities underneath
                st.caption(
                    f"P(cataract) = {p_cataract*100:.1f}%   |   "
                    f"P(normal) = {p_normal*100:.1f}%"
                )

                # Colour-coded risk display
                if "High risk" in risk:
                    risk_emoji = "🔴"
                elif "Medium" in risk:
                    risk_emoji = "🟠"
                elif "Very low" in risk:
                    risk_emoji = "🟢"
                else:
                    risk_emoji = "🟢"

                st.markdown(f"**Overall risk category:** {risk_emoji} **{risk}**")

                # =============================
                #  Advice section
                # =============================
                if p_cataract > 0.50:
                    st.markdown(
                        """
                        ### ✅ Medical Advice  
                        🔴 **Your results indicate more than a 50% likelihood of cataracts.**

                        Cataracts are:
                        - **Treatable**
                        - Often **reversible** with a short, safe outpatient surgery
                        - One of the **most successful procedures** in modern medicine  

                        We strongly recommend booking an appointment with an eye specialist
                        (ophthalmologist) for a full examination and discussion of treatment options.
                        """
                    )
                else:
                    st.markdown(
                        """
                        ### 🟢 General Advice  
                        Your screening suggests a **low likelihood of cataracts**.

                        However:
                        - Regular eye exams are still recommended  
                        - Cataracts and other eye conditions can develop with age  
                        - If you notice any changes in your vision, seek professional advice  
                        """
                    )

                st.markdown("### Patient factors used")
                st.write(
                    f"- Age: **{age}** years\n"
                    f"- Sex: **{sex}**\n"
                    f"- Smoker: **{smoker_str}**\n"
                    f"- Alcohol: **{alcohol_str}**"
                )

                st.markdown("### How to interpret this")
                st.markdown(
                    """
                    - The **percentage** above shows what the model is most confident about  
                      (either cataract or no cataract).
                    - The **risk category** combines the image with age, sex, smoking, and alcohol.  
                    - This is a prototype and **not a substitute** for a clinical diagnosis.
                    """
                )


if __name__ == "__main__":
    main()
