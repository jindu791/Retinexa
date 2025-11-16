import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models\cataract_resnet18_binary.pth"
VAL_DIR = "valid"   # folder containing /cataract and /normal

# -----------------------------------------------
# Load model
# -----------------------------------------------
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return model, transform


# -----------------------------------------------
# Evaluate with probabilities
# -----------------------------------------------
def evaluate():
    model, transform = load_model()

    dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    print("\nRunning confidence-based evaluation...\n")

    for img_tensor, label in loader:
        img_tensor = img_tensor.to(DEVICE)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1)[0]

        p_cataract = probs[0].item()
        p_normal = probs[1].item()

        pred = torch.argmax(probs).item()

        print("----------------------------------------")
        print(f"True Label    : {idx_to_class[label.item()]}")
        print(f"Predicted     : {idx_to_class[pred]}")
        print(f"Confidence    : {probs[pred].item():.4f}")
        print(f"(P(cataract)={p_cataract:.4f},  P(normal)={p_normal:.4f})")

    print("\nDone.\n")


if __name__ == "__main__":
    evaluate()
