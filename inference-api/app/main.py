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

    # ======================================================
    # ETAPA 1: DETECÇÃO (Normal vs Câncer) - Random Forest
    # ======================================================
    rf_features = preprocess_image_rf(image)
    rf_result = run_inference_rf(models['rf'], rf_features)
    
    # Prepara dados iniciais do JSON
    final_json = {
        "uuid": img_id,
        "s3_raw": raw_key,
        "s3_processed": None,
        "etapa_deteccao": rf_result,
        "etapa_classificacao": None, # Será preenchido se for cancer
        "resultado_final": rf_result["classe_detectada"]
    }

    processed_key = None
    treated_base64 = None

    # Se detectar CANCER, executamos a ResNet
    if rf_result["classe_detectada"] == "CANCER":
        
        # ======================================================
        # ETAPA 2: CLASSIFICAÇÃO (Benigno vs Maligno) - ResNet
        # ======================================================
        tensor, treated_pil = preprocess_image_resnet(image)
        
        # Upload da imagem processada (apenas se for relevante para visualização)
        processed_key = upload_processed_to_s3(img_id, treated_pil)
        treated_base64 = pil_to_base64(treated_pil)
        
        resnet_result = run_inference_resnet(models['resnet'], tensor)
        
        final_json["etapa_classificacao"] = resnet_result
        final_json["s3_processed"] = processed_key
        final_json["imagem_tratada_base64"] = treated_base64
        
        # Atualiza o resultado final para ser mais específico
        final_json["resultado_final"] = resnet_result["subtipo"] # BENIGNO ou MALIGNO

    # Se for NORMAL, não precisamos rodar a ResNet
    
    # Salvar resultado final no S3
    upload_results_json_to_s3(img_id, final_json)

    return final_json

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": {k: (v is not None) for k, v in models.items()}
    }
