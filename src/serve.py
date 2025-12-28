import torch
import gradio as gr
from PIL import Image
from torchvision import transforms

from config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, MODEL_DIR
from model import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = MODEL_DIR / "best_model.pth"
model, artist_names = None, []

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE + 36),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


def load_model_if_needed():
    global model, artist_names
    if model is None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")
        model, artist_names = load_model(model_path, device)
    return model, artist_names


def predict(image):
    if image is None:
        return {}
    
    try:
        model, artist_names = load_model_if_needed()
    except FileNotFoundError as e:
        return {"Error": str(e)}
    
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGB")
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    
    top_k = min(5, len(artist_names))
    top_probs, top_indices = torch.topk(probs, top_k)
    
    results = {artist_names[idx]: float(prob) for prob, idx in zip(top_probs, top_indices)}
    return results


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Artwork"),
    outputs=gr.Label(num_top_classes=5, label="Artist Predictions"),
    title="🎨 Art Style Classifier",
    description="Upload an artwork to predict which famous artist created it.",
    examples=[],
    allow_flagging="never",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch(share=False)
