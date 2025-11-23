from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os
from app.model_loader import load_models

from .s3_utils import upload_raw_to_s3, upload_processed_to_s3, upload_results_json_to_s3
from .utils import pil_to_base64
from .preprocess import load_image_any_format, preprocess_image_resnet, preprocess_image_rf
from .inference import run_inference_resnet, run_inference_rf

app = FastAPI(title="Mamografia Inference API (Pipeline Completo)", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carrega ambos os modelos na inicialização
models = load_models()

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

    # 1. Upload RAW para S3
    raw_key = upload_raw_to_s3(img_id, img_bytes, file.content_type)

    # 2. Carregar Imagem (PIL)
    try:
        image = load_image_any_format(img_bytes, ext)
    except Exception as e:
        raise HTTPException(400, f"Erro ao ler imagem: {str(e)}")

    # Base64 da imagem original (sempre disponível)
    original_base64 = pil_to_base64(image)

    # ======================================================
    # ETAPA 1: DETECÇÃO (Normal vs Câncer) - Random Forest
    # ======================================================
    rf_features = preprocess_image_rf(image)
    rf_result = run_inference_rf(models['rf'], rf_features)
    
    # Prepara dados iniciais do JSON
    final_json = {
        "uuid": img_id,

        "detect": {
            "classe": rf_result["classe_detectada"],
            "prob_normal": rf_result["prob_normal"],
            "prob_cancer": rf_result["prob_cancer"]
        },

        "classify": None,  # preenchido abaixo, se for CÂNCER

        "resultado_final": rf_result["classe_detectada"],

        "imagem_tratada_base64": None,  # sempre preenchido

        "s3": {
            "raw": raw_key,
            "processed": None,
            "results_json": None
        }
    }

    # =======================================
    # CASO 1 → CÂNCER: roda ResNet
    # =======================================
    if rf_result["classe_detectada"] == "CANCER":
        
        tensor, treated_pil = preprocess_image_resnet(image)

        processed_key = upload_processed_to_s3(img_id, treated_pil)
        final_json["s3"]["processed"] = processed_key

        # Base64 da imagem tratada
        treated_base64 = pil_to_base64(treated_pil)
        final_json["imagem_tratada_base64"] = treated_base64

        # Classificação final
        resnet_result = run_inference_resnet(models['resnet'], tensor)

        final_json["classify"] = {
            "subtipo": resnet_result["subtipo"],
            "prob_benigno": resnet_result["prob_benigno"],
            "prob_maligno": resnet_result["prob_maligno"],
            "confianca_subtipo": resnet_result["confianca_subtipo"]
        }

        final_json["resultado_final"] = resnet_result["subtipo"]

    # =======================================
    # CASO 2 → NORMAL: retorna base64 da original
    # =======================================
    else:
        final_json["imagem_tratada_base64"] = original_base64

    # Upload JSON final ao S3
    results_key = upload_results_json_to_s3(img_id, final_json)
    final_json["s3"]["results_json"] = results_key

    return final_json


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": {k: (v is not None) for k, v in models.items()}
    }
