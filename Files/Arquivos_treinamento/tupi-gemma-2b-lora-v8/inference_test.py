# Exemplo para carregar e testar - MÉTODO ALTERNATIVO
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import torch

base_model_id = "google/gemma-2b-it"
adapter_path = "./tupi-gemma-2b-lora-v8" # <<< SEU DIRETÓRIO DE SAÍDA
device = "cuda" if torch.cuda.is_available() else "cpu"
INSTRUCTION_TEXT = "Qual a definição em português da palavra Tupi a seguir?"

print(f"Carregando tokenizer de {adapter_path}...")
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Carregando modelo base E adaptadores LoRA de {adapter_path}...")

# Configuração de quantização (necessária se treinou com QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, # Use o mesmo dtype do treino
    bnb_4bit_use_double_quant=True,
)

# Carrega a configuração do adaptador para obter o ID do modelo base correto (opcional mas bom)
# config = PeftConfig.from_pretrained(adapter_path)

# Carrega o modelo base JÁ COM A QUANTIZAÇÃO
model = AutoModelForCausalLM.from_pretrained(
    base_model_id, # Usa o ID do modelo base original
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 # Definir dtype aqui também
)

# Carrega os pesos do adaptador NO MODELO JÁ CARREGADO
# IMPORTANTE: Usar o objeto 'model' carregado acima, não o 'base_model_id' novamente
model = PeftModel.from_pretrained(model, adapter_path) 

# Opcional: Mesclar para inferência (pode exigir mais RAM/VRAM temporariamente)
# print("Mesclando adaptadores...")
# model = model.merge_and_unload()

model.eval() # Coloca em modo de avaliação
print("Modelo pronto para inferência.")

# --- Função de Teste (mantida como antes) ---
def gerar_definicao(palavra_tupi):
    prompt = f"### Instruction:\n{INSTRUCTION_TEXT}\n\n### Input:\n{palavra_tupi}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(device)
    print(f"\n--- Gerando para: {palavra_tupi} ---")
    with torch.no_grad(): # Desabilita cálculo de gradientes para inferência
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    resultado_completo = tokenizer.decode(outputs[0], skip_special_tokens=True)
    partes = resultado_completo.split("### Response:")
    if len(partes) > 1:
        resposta = partes[-1].strip()
    else:
        resposta = resultado_completo.replace(prompt.replace("### Response:\n", ""), "").strip()
    print(f"Resposta Gerada:\n{resposta}")
    return resposta

# --- Teste com Palavras ---
palavras_teste = ["aba", "kunha", "y", "oka", "îandé", "pirá", "morubixaba"]
for palavra in palavras_teste:
    gerar_definicao(palavra)

# --- Teste Interativo (Opcional) ---
print("\n--- Teste Interativo (digite 'sair' para terminar) ---")
while True:
    input_word = input("Digite uma palavra Tupi: ")
    if input_word.lower() == 'sair': break
    if input_word: gerar_definicao(input_word)