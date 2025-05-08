import torch
from PIL import Image
from urllib.request import urlopen
from open_clip import create_model_from_pretrained, get_tokenizer

# Load model and preprocessing
model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Load image and preprocess
image = Image.open("/content/Screen Shot 2025-05-07 at 5.27.26 PM.png").convert("RGB")
image_input = preprocess(image).unsqueeze(0).to(device)

# Define multiple possible labels
labels = [
    "normal chest X-ray",
    "chest X-ray showing pneumonia",
    "chest X-ray with pleural effusion",
    "lung opacity in chest X-ray",
    "cardiomegaly visible in chest X-ray"
]

# Tokenize all prompts
text_inputs = tokenizer(labels).to(device)

# Inference
with torch.no_grad():
    image_features, text_features, logit_scale = model(image_input, text_inputs)
    # Compute raw cosine similarity
    similarities = image_features @ text_features.T
    probs = similarities.softmax(dim=-1).squeeze().cpu().numpy()

# Show ranked results
for label, score in sorted(zip(labels, probs), key=lambda x: x[1], reverse=True):
    print(f"{label}: {score:.4f}")
