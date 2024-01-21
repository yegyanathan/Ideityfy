import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
from module import myModule

CLASS_TO_IDX = ['AMMAN', 'AYYAPPA', 'BHAIRAV', 'BRAHMA', 'BUDDHA', 'DURGA', 'GANESHA', 'HANUMAN', 'KAALI', 
                'KRISHNA', 'KURMA', 'LAKSHMI', 'LINGA', 'MATSYA', 'MURUGA', 'NARASIMHA', 'NATARAJA', 'PARASURAMA', 
                'RAMA', 'SARASWATI', 'SHIVA', 'THIRTHANKARA', 'VAMANA', 'VARAHA', 'VISHNU']

IMG_SIZE = (224, 224)
STATS = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# Define the transformation for the input image
TTA_TRANSFORM = T.Compose([
    T.Resize(IMG_SIZE),
    T.AutoAugment(),
    T.ToTensor(),
    T.Normalize(**STATS)
])


st.set_page_config(
    page_title="Identify the deity using Computer Vision.",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is an *extremely* cool app!"
    }
)


st.title(":sparkles: I:orange[deity]fy")
st.header("Discover the deity with a snap.")

model = myModule.load_from_checkpoint("checkpoints/vit_base_clip_rank4.ckpt")
model.to("cpu")
model.eval()


# Function to make predictions
def predict(image):
    # Load and preprocess the input image
    with Image.open(image).convert('RGB') as img:
        img_tensor = torch.stack([TTA_TRANSFORM(img) for img in [img for _ in range(10)]])
        img_tensor = torch.mean(img_tensor, dim=0).unsqueeze(0)

    # Make a prediction
    with torch.no_grad():
        logits = model(img_tensor)

    # Get the top 3 predictions and their probabilities
    probs = torch.softmax(logits, dim=1)
    topk = torch.topk(probs, k=3)
    values, indices = topk.values, topk.indices

    values = values.squeeze().cpu().numpy().tolist()
    indices = indices.cpu().squeeze().numpy().tolist()

    return values, indices


# Upload image through Streamlit
img = st.file_uploader(label='choose a file', type=['png', 'jpg', 'jpeg'], label_visibility="hidden")


if img is not None:

    # Make predictions when the user clicks the "Predict" button
    if st.button("Predict"):
        values, indices = predict(img)
        classes = [CLASS_TO_IDX[index] for index in indices]
        # Display the top 3 predictions as a bar chart
        st.bar_chart({label: prob for label, prob in zip(indices, values)}, color="#FFC101")
