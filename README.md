🧿 Retinexa — AI-Powered Cataract Screening
Early Cataract Detection Using Retinal Imaging + Patient Risk Factors

Retinexa is an AI-driven cataract screening tool that combines:

🩻 Deep learning–based retina image analysis

👤 Patient risk factors such as age, sex, smoking, and alcohol use

📊 Interpretable risk scoring

🖥️ A clean Streamlit interface for real-time screening

This project demonstrates how AI can assist clinicians and improve early detection of cataracts—one of the world’s leading causes of reversible blindness.

📌 Features
🧠 1. Retina Image Classifier (ResNet-18)

Binary classification: Cataract vs Normal

Fine-tuned using high-quality AMDNet23 fundus images

Temperature scaling for smoother probabilities

Balanced training with equal samples per class

📋 2. Patient Risk Factors Integrated

Model combines image probability with:

Age

Sex

Smoking status

Alcohol consumption

Using a custom risk-decision algorithm to classify into:

Low

Medium

High (refer)

📈 3. Confidence-Based Output

The system shows whichever probability is higher:

“🟢 Chance of NO cataract: 86%”

or

“🔴 Chance of cataract: 72%”

For clarity and user reassurance.

🔔 4. Medical Advice Layer

If the model predicts >50% likelihood of cataracts, the UI automatically displays:

Cataracts are reversible

Surgery is quick and effective

Suggestion to see an ophthalmologist

💻 5. Streamlit Web App

A simple UI allowing:

Image upload

Entering patient details

Instant model inference

Full risk report

🚀 Demo Screenshot

(Add your app screenshot here)
![App Screenshot](screenshot.png)

📂 Project Structure
final_biotech/
│
├── app.py                          # Streamlit UI
├── train.py                        # Model training script
├── test.py                         # Model evaluation script
│
├── models/
│   └── cataract_resnet18_binary.pth    # Saved trained model
│
├── dataset/
│   ├── train/
│   │   ├── cataract/
│   │   └── normal/
│   └── valid/
│       ├── cataract/
│       └── normal/
│
├── retinexa.png                    # App logo
└── README.md

🛠️ Installation
1️⃣ Clone the repo
git clone https://github.com/yourusername/retinexa.git
cd retinexa

2️⃣ Install dependencies
pip install -r requirements.txt


Or install individually:

pip install torch torchvision streamlit pillow numpy scikit-learn

🎯 Usage
▶️ Run the app
streamlit run app.py

📊 Train the model
python train.py

🔍 Test on validation dataset
python test.py

🧪 Model Performance

Using the AMDNet23 dataset:

Metric	Score
Train Accuracy	~97–99%
Validation Accuracy	100%
Zero cataract misclassifications	✔️
Zero normal misclassifications	✔️

Confusion matrix:

	Pred Cataract	Pred Normal
True Cataract	100	0
True Normal	0	100
