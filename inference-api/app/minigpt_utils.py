import torch
import re
from pathlib import Path

# Tentativa de import do BaseGPTv02
try:
    from app.BaseGPTv02 import ModelConfig, Logger, TextDataProcessor, MiniGPT, TextGenerator
    minigpt_config = ModelConfig()
    minigpt_logger = Logger("MiniGPT_Pipeline")
except Exception as e:
    print("❌ Falha ao importar BaseGPTv02:", e)
    minigpt_config = None
    minigpt_logger = None
    TextDataProcessor = None
    MiniGPT = None
    TextGenerator = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH_MINIGPT = "minigpt_weights.pth"

CORPUS_MAP = {}
minigpt_generator = None
minigpt_processor = None
minigpt_loaded = False


# ---------------------------------------------------------
# 1. CARREGA O CORPUS (fallback + definição das faixas)
# ---------------------------------------------------------
def load_corpus():
    global CORPUS_MAP

    if not minigpt_config:
        print("MiniGPT indisponível — corpus não carregado.")
        return

    corpus_file = minigpt_config.input_file

    try:
        with open(corpus_file, "r", encoding="utf-8") as f:
            content = f.read()

        pattern = r"Probabilidade\s+(.*?)\s+(\d+)\%\s*-\s*(\d+)\%[\s\S]*?Recomendação:\s*([\s\S]*?)(?=Probabilidade|Fonte:|$)"
        matches = re.findall(pattern, content, re.IGNORECASE)

        for class_label, min_p, max_p, rec in matches:
            key = (class_label.upper().strip(), int(min_p), int(max_p))
            rec_clean = rec.strip().replace("\n", " ").replace("\\", "")
            CORPUS_MAP[key] = rec_clean

        print(f"📚 Corpus carregado com {len(CORPUS_MAP)} faixas.")

    except Exception as e:
        print("❌ Falha ao carregar corpus:", e)


# ---------------------------------------------------------
# 2. CONFIGURA O MiniGPT
# ---------------------------------------------------------
def load_minigpt():
    global minigpt_processor, minigpt_generator, minigpt_loaded

    if not TextDataProcessor:
        print("MiniGPT não disponível.")
        return

    try:
        # Carrega vocabulário
        minigpt_processor = TextDataProcessor(minigpt_config, minigpt_logger)
        minigpt_processor.load_and_process_data()

        # Cria modelo
        model = MiniGPT(minigpt_config, minigpt_processor.vocab_size).to(DEVICE)

        if Path(MODEL_PATH_MINIGPT).exists():
            model.load_state_dict(torch.load(MODEL_PATH_MINIGPT, map_location=DEVICE))
            model.eval()
            minigpt_loaded = True
            print("🤖 MiniGPT carregado!")
        else:
            print("⚠ Pesos MiniGPT não encontrados. Usando fallback.")

        minigpt_generator = TextGenerator(model, minigpt_processor, minigpt_config, minigpt_logger)

    except Exception as e:
        print("❌ Falha ao configurar MiniGPT:", e)
        minigpt_loaded = False


# ---------------------------------------------------------
# 3. GERA A RECOMENDAÇÃO
# ---------------------------------------------------------
def generate_recommendation(classe: str, confianca: float):
    classe = classe.upper()
    prob = round(confianca)

    # Encontra faixa correta
    chosen_key = None
    for (label, low, high), rec in CORPUS_MAP.items():
        if label == classe and low <= prob <= high:
            chosen_key = (label, low, high)
            break

    if not chosen_key:
        return "Recomendação não encontrada para este nível de confiança."

    prompt = f"Probabilidade {classe} {chosen_key[1]}% - {chosen_key[2]}% Recomendação:"

    # MiniGPT REAL
    if minigpt_loaded and minigpt_generator:
        try:
            gen = minigpt_generator.generate_text(
                prompt, max_new_tokens=200, temperature=0.7
            )
            if gen.startswith(prompt):
                gen = gen[len(prompt):]
            return gen.strip()
        except Exception:
            return CORPUS_MAP[chosen_key]

    # Fallback
    return CORPUS_MAP[chosen_key]


# Carrega tudo ao importar módulo
load_corpus()
load_minigpt()

