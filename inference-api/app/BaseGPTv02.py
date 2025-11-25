"""
MiniGPT: Implementação de um modelo Transformer simplificado para geração de texto
Versão 2.1 - Treinamento Focado em Recomendações (Diagnóstico e Porcentagem)

Autor: MR Autoral
Data: 2025
Versão: 2.1.1 - Correção NameError e Salvamento de Pesos
"""

import os
import logging
import re
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

# ============================================================================
# CONFIGURAÇÕES E HIPERPARÂMETROS
# ============================================================================

@dataclass
class ModelConfig:
    """Configurações do modelo MiniGPT"""
    batch_size: int = 32
    block_size: int = 64
    max_iters: int = 3000
    eval_interval: int = 300
    learning_rate: float = 3e-4
    eval_iters: int = 200
    n_embd: int = 128
    n_head: int = 8
    n_layer: int = 6
    dropout: float = 0.1
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 1337
    train_split: float = 0.9
    input_file: str = os.path.join(os.path.dirname(__file__), "corpus_tech.txt")


class Logger:
    """Classe para logging estruturado"""
    def __init__(self, name: str = "MiniGPT"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def info(self, message: str) -> None:
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        self.logger.error(message)

# ============================================================================
# PREPARAÇÃO E PROCESSAMENTO DE DADOS (LÓGICA DE TREINAMENTO FOCADA)
# ============================================================================

class TextDataProcessor:
    """Classe responsável pelo processamento de dados de texto"""
    
    def __init__(self, config: ModelConfig, logger: Logger):
        self.config = config
        self.logger = logger
        self.chars = []
        self.vocab_size = 0
        self.stoi = {}
        self.itos = {}
        self._is_initialized = False
        self.FULL_TEXT = ""
        
    def _build_vocabulary(self) -> None:
        """Constrói o vocabulário a partir do texto completo"""
        if not self.FULL_TEXT:
             raise ValueError("FULL_TEXT está vazio. Carregue os dados primeiro.")
             
        self.chars = sorted(list(set(self.FULL_TEXT)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        
        self._is_initialized = True
        self.logger.info(f"Vocabulário criado com {self.vocab_size} caracteres únicos")
    
    def _get_training_segments(self, content: str) -> str:
        """Extrai e formata os segmentos de treinamento (Diagnóstico + Recomendação)"""
        # Regex para capturar (HEADER - Probabilidade...) e (CORPO - Recomendação:...)
        pattern = r"(Probabilidade\s+[^\n]+)[\s\S]*?(Recomendação:[\s\S]*?)(?=Probabilidade|Fonte:|$)"
        
        matches = re.findall(pattern, content, re.IGNORECASE)
        segments = []
        
        for header, recommendation_block in matches:
            # CORREÇÃO AQUI: Remove tags de fonte e tags genéricas
            clean_rec = re.sub(r'\\', '', recommendation_block).strip()
            clean_rec = re.sub(r'\\', '', clean_rec).strip()
            
            # Formato de treinamento: [HEADER] [RECOMENDACAO LIMPA]
            segments.append(f"{header.strip()} {clean_rec.strip()}\n\n")
            
        return "".join(segments)

    def load_and_process_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Carrega, formata e processa os dados de texto do arquivo de entrada"""
        
        file_path = Path(self.config.input_file)
        if not file_path.exists():
            self.logger.error(f"❌ Erro Fatal: Arquivo de corpus '{self.config.input_file}' não encontrado.")
            raise FileNotFoundError(f"Arquivo de corpus não encontrado: {self.config.input_file}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                full_content = f.read()
        except Exception as e:
            self.logger.error(f"Erro ao carregar arquivo: {e}")
            raise
        
        if not full_content.strip():
            raise ValueError(f"Arquivo de texto '{self.config.input_file}' está vazio.")
        
        self.FULL_TEXT = full_content
        self.logger.info(f"Texto COMPLETO carregado: {len(full_content)} caracteres")
        
        self._build_vocabulary()
        training_text = self._get_training_segments(full_content)
        
        if not training_text.strip():
            raise ValueError("Não foi possível extrair segmentos de treinamento estruturados.")
            
        self.logger.info(f"Texto FORMATADO para Treino: {len(training_text)} caracteres (Segmentos: {training_text.count('Probabilidade')})")
        
        try:
            data = torch.tensor(self.encode(training_text), dtype=torch.long)
        except Exception as e:
            self.logger.error(f"Erro na tokenização: {e}")
            raise
        
        n = int(self.config.train_split * len(data))
        train_data = data[:n]
        val_data = data[n:]
        
        self.logger.info(f"Dados tokenizados divididos: {len(train_data)} treino, {len(val_data)} validação")
        
        return train_data, val_data
    
    def encode(self, text: str) -> list:
        if not self._is_initialized:
            raise RuntimeError("Vocabulário não foi inicializado.")
        return [self.stoi[c] for c in text if c in self.stoi]
    
    def decode(self, tokens: list) -> str:
        if not self._is_initialized:
            raise RuntimeError("Vocabulário não foi inicializado.")
        return ''.join([self.itos[i] for i in tokens if i in self.itos])

# ============================================================================
# DATA LOADER (ESSENCIAL - CORRIGE O NAMERROR)
# ============================================================================

class DataLoader:
    """Classe para carregamento de lotes de dados"""
    
    def __init__(self, train_data: torch.Tensor, val_data: torch.Tensor, 
                 config: ModelConfig):
        self.train_data = train_data
        self.val_data = val_data
        self.config = config
        
        if len(train_data) < config.block_size or len(val_data) < config.block_size:
            self.config.block_size = min(len(train_data), len(val_data)) // 2
            if self.config.block_size < 1:
                 raise ValueError(f"Dados insuficientes para treinamento.")
            self.config.batch_size = min(self.config.batch_size, len(train_data) // self.config.block_size)

    
    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Carrega um lote de dados"""
        data = self.train_data if split == 'train' else self.val_data
        
        max_start_idx = len(data) - self.config.block_size
        if max_start_idx <= 0:
            # Caso raro, mas protege contra falhas
            raise ValueError(f"Dados insuficientes para block_size {self.config.block_size}")
        
        ix = torch.randint(max_start_idx, (self.config.batch_size,))
        x = torch.stack([data[i:i+self.config.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.config.block_size+1] for i in ix])
        
        x, y = x.to(self.config.device), y.to(self.config.device)
        return x, y

# ============================================================================
# COMPONENTES DO MODELO TRANSFORMER (RESTO DO CÓDIGO)
# ============================================================================

class AttentionHead(nn.Module):
    # ... (Conteúdo da classe AttentionHead)
    def __init__(self, config: ModelConfig, head_size: int):
        super().__init__()
        self.head_size = head_size
        self.config = config
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer(
            'tril', 
            torch.tril(torch.ones(config.block_size, config.block_size))
        )
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    # ... (Conteúdo da classe MultiHeadAttention)
    def __init__(self, config: ModelConfig):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.heads = nn.ModuleList([
            AttentionHead(config, head_size) for _ in range(config.n_head)
        ])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    # ... (Conteúdo da classe FeedForward)
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    # ... (Conteúdo da classe TransformerBlock)
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# ============================================================================
# MODELO PRINCIPAL
# ============================================================================

class MiniGPT(nn.Module):
    # ... (Conteúdo da classe MiniGPT)
    def __init__(self, config: ModelConfig, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        if T > self.config.block_size:
            # Protege contra T maior que o block_size, mas permite inferência em T < block_size
             idx = idx[:, -self.config.block_size:]
             T = idx.shape[1]
             
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.config.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        
        return logits, loss
    
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.config.block_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# TREINADOR DO MODELO
# ============================================================================

class ModelTrainer:
    # ... (Conteúdo da classe ModelTrainer)
    def __init__(self, model: MiniGPT, data_loader: DataLoader, 
                 config: ModelConfig, logger: Logger):
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.logger = logger
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate
        )
        self.training_history = {'train_loss': [], 'val_loss': [], 'iterations': []}
    
    @torch.no_grad()
    def estimate_loss(self) -> Dict[str, float]:
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                try:
                    X, Y = self.data_loader.get_batch(split)
                    _, loss = self.model(X, Y)
                    losses[k] = loss.item()
                except Exception:
                    losses[k] = float('inf')
            out[split] = losses.mean().item()
        self.model.train()
        return out
    
    def train(self) -> Dict[str, Any]:
        self.logger.info("Iniciando treinamento...")
        for iter_num in range(self.config.max_iters):
            if iter_num % self.config.eval_interval == 0 or iter_num == self.config.max_iters - 1:
                losses = self.estimate_loss()
                self.logger.info(
                    f"Iteração {iter_num}: perda treino {losses['train']:.4f}, perda validação {losses['val']:.4f}"
                )
                self.training_history['train_loss'].append(losses['train'])
                self.training_history['val_loss'].append(losses['val'])
                self.training_history['iterations'].append(iter_num)
            
            try:
                xb, yb = self.data_loader.get_batch('train')
                _, loss = self.model(xb, yb)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
            except Exception as e:
                self.logger.error(f"Erro no passo de treinamento {iter_num}: {e}")
                continue
        self.logger.info("Treinamento concluído!")
        return self.training_history

# ============================================================================
# GERADOR DE TEXTO
# ============================================================================

class TextGenerator:
    """Classe para geração de texto"""
    
    def __init__(self, model: MiniGPT, processor: TextDataProcessor, 
                 config: ModelConfig, logger: Logger):
        self.model = model
        self.processor = processor
        self.config = config
        self.logger = logger
    
    def generate_text(self, prompt: str, max_new_tokens: int = 300, temperature: float = 0.8) -> str:
        """Gera texto a partir de um prompt"""
        
        # Codifica o prompt
        encoded_prompt = self.processor.encode(prompt)
        if not encoded_prompt:
            self.logger.warning("Prompt vazio após codificação, usando 'A'")
            encoded_prompt = self.processor.encode("A")
            
        context = torch.tensor(
            [encoded_prompt], 
            dtype=torch.long, 
            device=self.config.device
        )
        
        # Gera tokens
        generated_tokens = self.model.generate(
            context, 
            max_new_tokens, 
            temperature=temperature
        )[0].tolist()
        
        # Decodifica para texto
        generated_text = self.processor.decode(generated_tokens)
        
        return generated_text

# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """Função principal para execução do MiniGPT"""
    try:
        # Configuração
        config = ModelConfig()
        logger = Logger()
        
        logger.info("🚀 Iniciando MiniGPT v2.1 (Treinamento Focado)...")
        torch.manual_seed(config.seed)
        
        # Processamento de dados
        logger.info("📚 Carregando e processando dados...")
        processor = TextDataProcessor(config, logger)
        train_data, val_data = processor.load_and_process_data()
        
        # ❌ CORREÇÃO: DataLoader é definido acima e agora pode ser chamado
        data_loader = DataLoader(train_data, val_data, config)
        
        # Criação do modelo
        logger.info("🧠 Criando modelo...")
        model = MiniGPT(config, processor.vocab_size)
        model = model.to(config.device)
        
        # Treinamento
        logger.info("🎯 Iniciando treinamento...")
        trainer = ModelTrainer(model, data_loader, config, logger)
        training_history = trainer.train()
        
        # Geração de texto (Exemplos)
        generator = TextGenerator(model, processor, config, logger)
        
        # Prompts ALINHADOS com o novo formato de treinamento
        prompts = [
            "Probabilidade Benigno 50% - 75% Recomendação:",
            "Probabilidade Maligno 50% - 75% Recomendação:",
            "Probabilidade Benigno 76% - 100% Recomendação:",
            "Probabilidade Maligno 76% - 100% Recomendação:",
        ]
        
        logger.info("\n🎨 === EXEMPLOS DE GERAÇÃO DE TEXTO FOCADO ===")
        for i, prompt in enumerate(prompts, 1):
             # ... (Código de geração de exemplos) ...
             try:
                generated_text = generator.generate_text(
                    prompt, 
                    max_new_tokens=200, 
                    temperature=0.8
                )
                print(f"\n{'='*80}")
                print(f"🔤 Prompt: '{prompt}'")
                print(f"{'─'*80}")
                print(f"🤖 Texto Gerado:")
                print(generated_text)
                print(f"{'='*80}")
             except Exception as e:
                logger.error(f"Erro ao gerar texto para prompt '{prompt}': {e}")


        # Estatísticas finais e SALVAMENTO
        final_train_loss = training_history['train_loss'][-1] if training_history['train_loss'] else 'N/A'
        logger.info(f"\n📊 === ESTATÍSTICAS FINAIS ===")
        logger.info(f"📈 Perda final de treino: {final_train_loss}")
        
        # 💾 SALVAR PESOS DO MODELO TREINADO (Crucial para o app.py)
        try:
            SAVE_PATH = 'minigpt_weights.pth'
            torch.save(model.state_dict(), SAVE_PATH)
            logger.info(f"💾 Pesos do MiniGPT salvos em {SAVE_PATH}")
        except Exception as e:
            logger.error(f"Erro ao salvar o modelo: {e}")
        
    except Exception as e:
        logger.error(f"❌ Erro fatal: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()