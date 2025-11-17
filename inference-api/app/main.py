from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import uuid
import io
import os
from app.model_loader import load_model


from .s3_utils import upload_raw_to_s3, upload_processed_to_s3, upload_results_json_to_s3
from .utils import pil_to_base64
from .preprocess import load_image_any_format, preprocess_image
from .inference import run_inference


app = FastAPI(title="Mamografia Inference API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model()


@app.post("/processar")
async def processar_imagem(file: UploadFile = File(...)):
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    allowed = {".png", ".jpg", ".jpeg", ".pgm", ".dcm"}

    if ext not in allowed:
        raise HTTPException(
            400,
            f"Formato '{ext}' não suportado. Use: {allowed}"
        )

    img_id = str(uuid.uuid4())
    img_bytes = await file.read()

    # Salvar RAW no S3
    raw_key = upload_raw_to_s3(img_id, img_bytes, file.content_type)

    # Carregar imagem (com DICOM)
    image = load_image_any_format(img_bytes, ext)

    # Tratamento → tensor + imagem tratada
    tensor, treated_pil = preprocess_image(image)

    # Converter imagem tratada para base64
    treated_base64 = pil_to_base64(treated_pil)

    # Salvar imagem tratada no S3
    processed_key = upload_processed_to_s3(img_id, treated_pil)

    # Inferência
    result = run_inference(model, tensor)

    # JSON final
    final_json = {
        "uuid": img_id,
        "classe": result["classe"],
        "prob_benigno": result["prob_benigno"],
        "prob_maligno": result["prob_maligno"],
        "confianca": result["confianca"],
        "imagem_tratada_base64": treated_base64,
        "s3_raw": raw_key,
        "s3_processed": processed_key
    }

    # Salvar resultado no S3
    upload_results_json_to_s3(img_id, final_json)
    
    upload_results_json_to_s3(img_id, final_json)

    return final_json

@app.get("/health")
def health():
    return {"status": "ok"}
