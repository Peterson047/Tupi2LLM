# Exemplo para carregar e testar
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_id = "google/gemma-2b-it"
adapter_path = "./tupi-gemma-2b-lora-v8" # <<< SEU DIRETÓRIO DE SAÍDA
device = "cuda" if torch.cuda.is_available() else "cpu"
INSTRUCTION_TEXT = "Qual a definição em português da palavra Tupi a seguir?" # Use a mesma instrução do treino

print(f"Carregando tokenizer de {adapter_path}...")
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token # Necessário para Gemma

print(f"Carregando modelo base {base_model_id}...")
# Carrega em bfloat16 para economizar memória, ajuste se necessário
# Se treinou com QLoRA, teoricamente não precisa da config de quantização aqui,
# mas pode ser necessário se for carregar o modelo base quantizado antes de aplicar adaptadores.
# Teste sem a config primeiro.
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto", # Tenta colocar na GPU
    trust_remote_code=True
)

print(f"Carregando adaptadores LoRA de {adapter_path}...")
# Carrega os adaptadores no modelo base
model = PeftModel.from_pretrained(base_model, adapter_path)

# Opcional, mas recomendado para inferência mais rápida se tiver memória:
# print("Mesclando adaptadores...")
# model = model.merge_and_unload()

model.eval() # Coloca em modo de avaliação
print("Modelo pronto para inferência.")

# --- Função de Teste ---
def gerar_definicao(palavra_tupi):
    prompt = f"### Instruction:\n{INSTRUCTION_TEXT}\n\n### Input:\n{palavra_tupi}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(device) # Pede attention_mask

    print(f"\n--- Gerando para: {palavra_tupi} ---")
    # Parâmetros de geração (ajuste conforme necessário)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,       # Limita o tamanho da resposta
        temperature=0.7,         # Controla a aleatoriedade (valores menores são mais focados)
        do_sample=True,          # Habilita amostragem para respostas mais variadas
        top_k=50,                # Considera as 50 palavras mais prováveis
        top_p=0.95,              # Considera palavras até somar 95% de probabilidade
        eos_token_id=tokenizer.eos_token_id, # Define o token de parada
        pad_token_id=tokenizer.pad_token_id  # Usa o pad_token_id
    )
    resultado_completo = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extrai apenas a resposta após "### Response:"
    partes = resultado_completo.split("### Response:")
    if len(partes) > 1:
        resposta = partes[-1].strip()
    else:
        # Se o marcador não foi gerado, tenta pegar o que veio depois do prompt
        resposta = resultado_completo.replace(prompt.replace("### Response:\n", ""), "").strip()

    print(f"Resposta Gerada:\n{resposta}")
    return resposta

# --- Teste com Palavras ---
palavras_teste = ["abá", "kunhã", "y", "oka", "îandé", "pirá", "morubixaba"] # Use palavras do seu dicionário
for palavra in palavras_teste:
    gerar_definicao(palavra)

# --- Teste Interativo (Opcional) ---
print("\n--- Teste Interativo (digite 'sair' para terminar) ---")
while True:
    input_word = input("Digite uma palavra Tupi: ")
    if input_word.lower() == 'sair':
        break
    if input_word:
        gerar_definicao(input_word)