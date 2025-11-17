import base64
import io
from PIL import Image

def pil_to_base64(pil_img: Image.Image):
    """Converte uma imagem PIL → base64 PNG."""
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")
