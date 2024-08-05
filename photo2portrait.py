import streamlit as st
import cv2
import os
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import io  # Ensure this import is present
from model.u2net import U2NET  # Replace with the actual import path

# Constants
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Preprocessing function
def preprocess_image(image):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

# Model loading function
def load_model(model, model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights file not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    return model

# Initialize U2Net model
u2net = U2NET(in_ch=3, out_ch=1)

# Load pre-trained model (replace with your model path)
try:
    u2net = load_model(u2net, "u2net_portrait.pth", device)
except FileNotFoundError as e:
    st.error(f"Error: {e}")
    st.stop()

# Function to detect the largest face in an image
def detect_single_face(face_cascade, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        st.warning("No face detected; processing the entire image.")
        return None

    # Filter to keep the largest face
    largest_face = max(faces, key=lambda b: b[2] * b[3])
    return largest_face

# Function to crop and resize face region
def crop_face(img, face):
    if face is None:
        return img

    (x, y, w, h) = face
    height, width = img.shape[0:2]

    # Define padding
    lpad = int(float(w) * 0.4)
    rpad = int(float(w) * 0.4)
    tpad = int(float(h) * 0.6)
    bpad = int(float(h) * 0.2)

    left, right = max(x - lpad, 0), min(x + w + rpad, width)
    top, bottom = max(y - tpad, 0), min(y + h + bpad, height)

    im_face = img[top:bottom, left:right]

    if len(im_face.shape) == 2:
        im_face = np.repeat(im_face[:, :, np.newaxis], 3, axis=2)

    # Pad to make image square
    hf, wf = im_face.shape[0:2]
    if hf > wf:
        wfp = (hf - wf) // 2
        im_face = np.pad(im_face, ((0, 0), (wfp, wfp), (0, 0)), mode='constant', constant_values=255)
    elif wf > hf:
        hfp = (wf - hf) // 2
        im_face = np.pad(im_face, ((hfp, hfp), (0, 0), (0, 0)), mode='constant', constant_values=255)

    im_face = cv2.resize(im_face, (512, 512), interpolation=cv2.INTER_AREA)
    return im_face

# Normalize prediction function
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    return (d - mi) / (ma - mi)

# Inference function
def inference(net, input):
    input = input / np.max(input)
    tmpImg = np.zeros((input.shape[0], input.shape[1], 3))
    tmpImg[:, :, 0] = (input[:, :, 2] - 0.406) / 0.225
    tmpImg[:, :, 1] = (input[:, :, 1] - 0.456) / 0.224
    tmpImg[:, :, 2] = (input[:, :, 0] - 0.485) / 0.229
    tmpImg = tmpImg.transpose((2, 0, 1))[np.newaxis, :, :, :]
    tmpImg = torch.from_numpy(tmpImg).type(torch.FloatTensor)

    if torch.cuda.is_available():
        tmpImg = tmpImg.cuda()

    with torch.no_grad():
        d1, _, _, _, _, _, _ = net(tmpImg)

    pred = 1.0 - d1[:, 0, :, :]
    pred = normPRED(pred)
    pred = pred.squeeze().cpu().data.numpy()
    return pred

# Convert image to pencil drawing
def image_to_pencil_drawing(image, line_size, line_density):
    img_cv = np.array(image)
    face = detect_single_face(face_cascade, img_cv)
    im_face = crop_face(img_cv, face)

    preprocessed_image = preprocess_image(Image.fromarray(im_face))

    with torch.no_grad():
        output = u2net(preprocessed_image)

    output_data = output[0].squeeze().cpu().numpy()
    pencil_drawing = Image.fromarray((output_data * 255).astype(np.uint8), mode='L')

    img_cv = cv2.cvtColor(np.array(pencil_drawing), cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face_roi = img_cv[y:y+h, x:x+w]

    pencil_drawing = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    pencil_drawing = pencil_drawing.convert('RGB')
    pencil_drawing = pencil_drawing.point(lambda p: p * line_density)

    # Apply Gaussian Blur to adjust line size
    pencil_drawing_array = np.array(pencil_drawing)
    blurred = cv2.GaussianBlur(pencil_drawing_array, (line_size, line_size), 0)
    inverted_image = 255 - blurred
    inverted_pencil_drawing = Image.fromarray(inverted_image.astype(np.uint8))

    return inverted_pencil_drawing

# Fix image function to handle default and uploaded images
def fix_image(upload=None):
    if upload:
        image = Image.open(upload)
    else:
        image = Image.open("8.jpg")  # Default image path

    # Create columns for side-by-side display
    col1, col2 = st.columns(2)

    # Display the original image
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process and display the pencil drawing
    pencil_drawing = image_to_pencil_drawing(image, line_size, line_density)

    with col2:
        st.image(pencil_drawing, caption="Pencil Drawing", use_column_width=True)

    # Provide download option for the pencil drawing
    buf = io.BytesIO()
    pencil_drawing.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.sidebar.download_button(
        label="Download Pencil Drawing",
        data=byte_im,
        file_name="pencil_drawing.png",
        mime="image/png"
    )

# Streamlit app setup
st.title("Image to Pencil Drawing")

# Sidebar for file upload and controls
st.sidebar.title("Controls :gear:")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

# Slider controls for line size and density
line_size = st.sidebar.slider("Line Size", min_value=1, max_value=15, value=5, step=2)
line_density = st.sidebar.slider("Line Density", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

# Determine image processing
if uploaded_file is not None:
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        fix_image(upload=uploaded_file)
else:
    fix_image()  # Use default image if none uploaded

# Add custom CSS for dark theme
st.markdown(
    """
    <style>
    body {
        background-color: #1E1E1E;  /* Dark background color */
        color: #FFFFFF;  /* White text color */
    }
    </style>
    """,
    unsafe_allow_html=True
)
