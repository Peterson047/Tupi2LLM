---
library_name: peft
license: gemma
base_model: google/gemma-2b-it
tags:
- generated_from_trainer
datasets:
- train_dataset.jsonl
model-index:
- name: tupi-gemma-2b-lora-v8
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/axolotl-ai-cloud/axolotl)
<details><summary>See axolotl config</summary>

axolotl version: `0.8.0`
```yaml
# Configuração Axolotl para Fine-tuning Tupi (GTX 1650 - 4GB VRAM)

# --- Modelo Base ---
base_model: google/gemma-2b-it   # Modelo pequeno e bom em instruções
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer
trust_remote_code: true          # Necessário para Gemma

# --- Datasets ---
datasets:
  - path: train_dataset.jsonl    # <<< AJUSTE O CAMINHO se necessário
    type: alpaca                 # Formato instruction/input/output é compatível com 'alpaca'
# Se você tiver pouca memória RAM (não VRAM), pode usar shards:
#   shards: 5                   # Divide o carregamento em partes (ajuste conforme RAM)

dataset_prepared_path: last_run_prepared_dataset # Onde salvar o dataset tokenizado


# --- Dataset de Validação ---
val_datasets:                    # Define explicitamente o arquivo de validação
  - path: validation_dataset.jsonl # <<< AJUSTE O CAMINHO se necessário
    type: alpaca

# --- Configurações de Saída e Treinamento ---
output_dir: ./tupi-gemma-2b-lora-v8 # << Onde salvar os adaptadores LoRA
sequence_len: 256             # << IMPORTANTE: Reduzido para caber em 4GB VRAM
sample_packing: true             # Tenta empacotar sequências para usar melhor o comprimento
pad_to_sequence_len: true        # Garante que todas as sequências tenham o mesmo tamanho (útil c/ packing)

# --- Otimização para Baixa VRAM (QLoRA) ---
load_in_4bit: true               # ESSENCIAL: Ativa quantização de 4 bits (QLoRA)
bnb_4bit_quant_type: nf4
bnb_4bit_compute_dtype: bfloat16 # Tenta usar bfloat16; se der erro, mude para float16
bnb_4bit_use_double_quant: true

# --- Configuração LoRA ---
adapter: lora
lora_r: 8                        # << Rank baixo para economizar memória
lora_alpha: 16                   # << Geralmente 2*r
lora_dropout: 0.05
lora_target_modules:             # Módulos específicos para Gemma 2B (verificar se são os ideais)
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
# Se der erro de módulo não encontrado, pode tentar deixar Axolotl detectar (remova ou comente a lista acima)
# lora_target_linear: true      # Alternativa para detectar automaticamente camadas lineares
lora_fan_in_fan_out: false
lora_bias: none

# --- Hiperparâmetros de Treinamento (Ajustados para baixa VRAM) ---
gradient_accumulation_steps: 8   # << Aumentado para compensar batch size pequeno
micro_batch_size: 1              # << Batch size por dispositivo MUITO PEQUENO (essencial para 4GB)
num_epochs: 1                    # << Comece com apenas 1 época para testar
optimizer: paged_adamw_8bit      # Otimizador eficiente em memória
lr_scheduler: cosine             # Programador de taxa de aprendizado
learning_rate: 1e-5              # << Taxa de aprendizado um pouco menor pode ser mais estável

# --- Outras Configurações ---
train_on_inputs: false           # Não treinar na instrução/input, apenas no output
group_by_length: false           # Pode economizar um pouco, mas desabilitado por segurança
bf16: auto
fp16: false # bf16 tem preferência se disponível (GPUs Ampere+)
tf32: false # Desabilitar tf32 pode economizar um pouco de memória em certas GPUs

gradient_checkpointing: true     # ESSENCIAL: Economiza muita VRAM
# early_stopping_patience: 3    # Desabilitado para o teste inicial
# resume_from_checkpoint: false # Não retomar de checkpoint
logging_steps: 10                # Log a cada 10 passos
eval_steps: 50                   # Avaliar a cada 50 passos (ajuste conforme tamanho do dataset)
save_steps: 100                  # Salvar checkpoint a cada 100 passos
save_total_limit: 1              # Manter apenas o último checkpoint
# eval_table_size: 100          # Opcional: Para logar exemplos de validação no W&B
# eval_max_new_tokens: 100      # Opcional: Limitar tokens gerados na avaliação

# --- W&B (Monitoramento Opcional) ---
# wandb_project: tupi-finetune
# wandb_run_id: gemma-2b-v8-run1
# wandb_log_model: false # Não salvar modelo no W&B por padrão

# --- Tokenizer ---
# Geralmente não precisa mexer se usar modelos do Hub
# special_tokens:
#   bos_token: "<s>"
#   eos_token: "</s>"
#   unk_token: "<unk>"
# pad_token: "</s>" # Gemma usa eos_token como pad_token
```

</details><br>

# tupi-gemma-2b-lora-v8

This model is a fine-tuned version of [google/gemma-2b-it](https://huggingface.co/google/gemma-2b-it) on the train_dataset.jsonl dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 8
- optimizer: Use paged_adamw_8bit with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 7
- num_epochs: 1.0

### Training results



### Framework versions

- PEFT 0.15.1
- Transformers 4.51.3
- Pytorch 2.5.1+cu124
- Datasets 3.5.0
- Tokenizers 0.21.1