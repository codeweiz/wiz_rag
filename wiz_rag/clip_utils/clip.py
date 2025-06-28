# 定义特征提取器
import clip
from PIL import Image


# load CLIP model
def get_clip_model():
    model_name = "ViT-B/32"
    model, preprocess = clip.load(model_name, device="cpu")
    model.eval()
    return model, preprocess


# define a function to encode images
def encode_image(image_path, model, preprocess):
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    image_features = model.encode_image(image)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.squeeze().tolist()


# define a function to encode text
def encode_text(text, model, preprocess):
    text_tokens = clip.tokenize(text)
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.squeeze().tolist()
