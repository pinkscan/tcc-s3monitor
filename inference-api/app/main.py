from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os

# =====================================================================
# IMPORTS DA PIPELINE EXISTENTE
# =====================================================================
from app.model_loader import load_models
from .s3_utils import upload_raw_to_s3, upload_processed_to_s3, upload_results_json_to_s3
from .utils import pil_to_base64
from .preprocess import load_image_any_format, preprocess_image_resnet, preprocess_image_rf
from .inference import run_inference_resnet, run_inference_rf

# =====================================================================
# IMPORT DO MINIGPT (usando módulo externo)
# =====================================================================
from app.minigpt_utils import generate_recommendation, minigpt_loaded, DEVICE

# =====================================================================
# CONFIG FASTAPI
# =====================================================================
app = FastAPI(title="Mamografia Inference API (Pipeline Completo)", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carrega ambos os modelos: RandomForest + ResNet
models = load_models()


# =====================================================================
# ROTA PRINCIPAL — PROCESSAMENTO COMPLETO
# =====================================================================
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

    # ======================================================
    # UPLOAD RAW PARA O S3
    # ======================================================
    raw_key = upload_raw_to_s3(img_id, img_bytes, file.content_type)

    # ======================================================
    # ABRIR IMAGEM
    # ======================================================
    try:
        image = load_image_any_format(img_bytes, ext)
    except Exception as e:
        raise HTTPException(400, f"Erro ao ler imagem: {str(e)}")

    original_base64 = pil_to_base64(image)

    # ======================================================
    # ETAPA 1: DETECÇÃO RANDOM FOREST
    # ======================================================
    rf_features = preprocess_image_rf(image)
    rf_result = run_inference_rf(models["rf"], rf_features)

    final_json = {
        "uuid": img_id,

        "detect": {
            "classe": rf_result["classe_detectada"],
            "prob_normal": rf_result["prob_normal"],
            "prob_cancer": rf_result["prob_cancer"]
        },

        "classify": None,
        "resultado_final": rf_result["classe_detectada"],
        "imagem_tratada_base64": None,

        "recomendacao": None,  # MiniGPT entra aqui

        "s3": {
            "raw": raw_key,
            "processed": None,
            "results_json": None
        }
    }

    # ======================================================
    # CASO 1 — CÂNCER → RODA RESNET + MINIGPT
    # ======================================================
    if rf_result["classe_detectada"] == "CANCER":

        tensor, treated_pil = preprocess_image_resnet(image)

        processed_key = upload_processed_to_s3(img_id, treated_pil)
        final_json["s3"]["processed"] = processed_key

        treated_base64 = pil_to_base64(treated_pil)
        final_json["imagem_tratada_base64"] = treated_base64

        resnet_result = run_inference_resnet(models["resnet"], tensor)

        final_json["classify"] = {
            "subtipo": resnet_result["subtipo"],
            "prob_benigno": resnet_result["prob_benigno"],
            "prob_maligno": resnet_result["prob_maligno"],
            "confianca_subtipo": resnet_result["confianca_subtipo"]
        }

        final_json["resultado_final"] = resnet_result["subtipo"]

        # ======================================================
        # RECOMENDAÇÃO COM MiniGPT
        # ======================================================
        classe_tex = "MALIGNO" if resnet_result["prob_maligno"] > resnet_result["prob_benigno"] else "BENIGNO"
        confianca = max(resnet_result["prob_maligno"], resnet_result["prob_benigno"])

        final_json["recomendacao"] = generate_recommendation(classe_tex, confianca)

    # ======================================================
    # CASO 2 — NORMAL
    # ======================================================
    else:
        final_json["imagem_tratada_base64"] = original_base64

        final_json["recomendacao"] = generate_recommendation(
            "BENIGNO",
            rf_result["prob_normal"]
        )

    # ======================================================
    # UPLOAD DO JSON FINAL PARA O S3
    # ======================================================
    results_key = upload_results_json_to_s3(img_id, final_json)
    final_json["s3"]["results_json"] = results_key

    return final_json


# =====================================================================
# HEALTHCHECK
# =====================================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": {k: (v is not None) for k, v in models.items()},
        "minigpt_loaded": minigpt_loaded,
        "device": DEVICE
    }
