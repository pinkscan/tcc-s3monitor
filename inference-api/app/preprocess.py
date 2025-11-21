from PIL import Image
import io
import torchvision.transforms as T
import numpy as np
import pydicom

IMG_SIZE_RESNET = 224
CROP_SIZE_RF = 64

# Transformações para o modelo ResNet
tensor_transform = T.Compose([
    T.Resize((IMG_SIZE_RESNET, IMG_SIZE_RESNET)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])


def load_image_any_format(image_bytes: bytes, file_ext: str):
    file_ext = file_ext.lower()

    # DICOM
    if file_ext == ".dcm":
        dcm = pydicom.dcmread(io.BytesIO(image_bytes))
        img_array = dcm.pixel_array

        # Normalização básica para visualização
        img_norm = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8) * 255
        image = Image.fromarray(img_norm.astype('uint8')).convert("RGB")
        return image

    # PNG/JPG/PGM etc
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def preprocess_image_resnet(image: Image.Image):
    """
    Prepara imagem para a ResNet (Classificação Benigno/Maligno)
    """
    image_resized = image.resize((IMG_SIZE_RESNET, IMG_SIZE_RESNET))
    tensor = tensor_transform(image_resized).unsqueeze(0)
    return tensor, image_resized


def preprocess_image_rf(image: Image.Image, crop_size=CROP_SIZE_RF):
    """
    Prepara imagem para o Random Forest (Detecção Normal/Cancer).
    Reimplementa a lógica de corte central e flatten do código Flask original.
    """
    # Converter para Grayscale (equivalente a cv2.IMREAD_GRAYSCALE)
    img_gray = image.convert("L")
    img_arr = np.array(img_gray)

    h, w = img_arr.shape
    cx, cy = w // 2, h // 2
    half = crop_size // 2

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, x1 + crop_size)
    y2 = min(h, y1 + crop_size)

    # Ajustes de borda
    if (x2 - x1) < crop_size:
        x1 = max(0, x2 - crop_size)
    if (y2 - y1) < crop_size:
        y1 = max(0, y2 - crop_size)

    crop = img_arr[y1:y2, x1:x2]

    # Resize se necessário (usando PIL para consistência ou cv2 se preferir, aqui manual via PIL)
    if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
        # Converter crop de volta pra PIL para redimensionar fácil
        crop_pil = Image.fromarray(crop)
        crop_pil = crop_pil.resize((crop_size, crop_size), Image.Resampling.BILINEAR)
        crop = np.array(crop_pil)

    # Normalização e Flatten
    crop = crop.astype(np.float32) / 255.0
    return crop.flatten().reshape(1, -1)
