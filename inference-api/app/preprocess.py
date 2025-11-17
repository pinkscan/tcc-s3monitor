from PIL import Image
import io
import torchvision.transforms as T
import numpy as np
import pydicom

IMG_SIZE = 224

# Transformações para o modelo
tensor_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
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

        img_norm = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8) * 255
        image = Image.fromarray(img_norm.astype('uint8')).convert("RGB")
        return image

    # PNG/JPG/etc
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")

    return image



def preprocess_image(image: Image.Image):
    """
    Retorna:
        - tensor pronto pro modelo (normalizado)
        - imagem PIL tratada (224x224) para enviar ao front
    """

    # criar versão visual (antes da normalização)
    image_resized = image.resize((IMG_SIZE, IMG_SIZE))

    # gerar tensor normalizado
    tensor = tensor_transform(image_resized).unsqueeze(0)

    return tensor, image_resized
