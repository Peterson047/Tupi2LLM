import json
import random
import os
from sklearn.model_selection import train_test_split # Biblioteca para divisão

# ==============================================================================
# --- Configurações ---
# ==============================================================================
print("--- Configurações da Divisão ---")

# 1. CAMINHO PARA O JSONL DE ENTRADA (O dataset de instrução gerado)
#    AJUSTE ESTE CAMINHO! Pode ser o v8_final ou v8_cleanup, dependendo do que você decidiu usar.
#    Se você copiou para dentro do WSL, use o caminho DENTRO do WSL (ex: '/root/tupi_project/dataset_instrucao_v8_final.jsonl')
caminho_jsonl_entrada = 'dataset_instrucao_v8_final.jsonl' # <<< VERIFIQUE E AJUSTE!

# 2. NOMES DOS ARQUIVOS DE SAÍDA
nome_arquivo_treino = 'train_dataset.jsonl'
nome_arquivo_valid = 'validation_dataset.jsonl'

# 3. PROPORÇÃO PARA VALIDAÇÃO (ex: 0.1 = 10%)
proporcao_validacao = 0.1

# 4. SEMENTE ALEATÓRIA (para resultados reproduzíveis)
random_seed = 42

# --- Determinar Diretório de Saída ---
# Salva os arquivos de treino/validação no mesmo diretório do arquivo de entrada
caminho_output_dir = os.path.dirname(caminho_jsonl_entrada)
if not caminho_output_dir: # Se o arquivo de entrada estiver no diretório atual
     caminho_output_dir = '.'

# Caminhos completos para os arquivos de saída
caminho_jsonl_treino = os.path.join(caminho_output_dir, nome_arquivo_treino)
caminho_jsonl_valid = os.path.join(caminho_output_dir, nome_arquivo_valid)

print(f"Arquivo de Entrada: {caminho_jsonl_entrada}")
print(f"Arquivo de Treino (Saída): {caminho_jsonl_treino}")
print(f"Arquivo de Validação (Saída): {caminho_jsonl_valid}")
print(f"Proporção de Validação: {proporcao_validacao:.1%}")

# ==============================================================================
# --- PASSO 1: Ler todas as linhas do JSONL de entrada ---
# ==============================================================================
print("\n--- PASSO 1: Lendo arquivo JSONL de entrada ---")
linhas_dados = []
try:
    with open(caminho_jsonl_entrada, 'r', encoding='utf-8') as f_in:
        for i, line in enumerate(f_in):
            line = line.strip()
            if line: # Ignora linhas em branco
                try:
                    linhas_dados.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Aviso: Erro ao decodificar JSON na linha {i+1}. Linha ignorada. Erro: {e}")
                    print(f"   Conteúdo da linha: {line[:100]}...")
                    continue
    print(f"Total de linhas de dados lidas: {len(linhas_dados)}")

except FileNotFoundError:
    print(f"Erro Crítico: Arquivo de entrada '{caminho_jsonl_entrada}' não encontrado.")
    exit()
except Exception as e:
    print(f"Erro Crítico ao ler o arquivo de entrada: {e}")
    exit()

if not linhas_dados:
    print("Erro Crítico: Nenhum dado válido foi lido do arquivo de entrada.")
    exit()
# Define um mínimo razoável para divisão
min_linhas_para_dividir = 10
if len(linhas_dados) < min_linhas_para_dividir:
    print(f"Erro: Dataset com menos de {min_linhas_para_dividir} linhas é muito pequeno para dividir em treino e validação.")
    exit()

# ==============================================================================
# --- PASSO 2: Dividir os dados ---
# ==============================================================================
print("\n--- PASSO 2: Dividindo os dados em treino e validação ---")
try:
    # Usar train_test_split para divisão aleatória
    train_data, val_data = train_test_split(
        linhas_dados,
        test_size=proporcao_validacao,
        random_state=random_seed,
        shuffle=True # Embaralhar os dados é importante
    )
    # Garante que mesmo com proporção pequena, tenhamos pelo menos 1 item na validação se possível
    if len(val_data) == 0 and len(train_data) > 0:
        print("Aviso: Conjunto de validação ficou vazio devido ao tamanho pequeno/proporção. Movendo uma amostra do treino.")
        val_data.append(train_data.pop())

    print(f"Tamanho do conjunto de treino: {len(train_data)} ({len(train_data)/len(linhas_dados):.1%})")
    print(f"Tamanho do conjunto de validação: {len(val_data)} ({len(val_data)/len(linhas_dados):.1%})")

except ValueError as e:
    print(f"Erro durante a divisão dos dados: {e}")
    print("Verifique se a proporção de validação é válida (entre 0.0 e 1.0) e se há dados suficientes.")
    exit()
except Exception as e:
    print(f"Erro inesperado durante a divisão dos dados: {e}")
    exit()

# ==============================================================================
# --- PASSO 3: Salvar os arquivos JSONL de saída ---
# ==============================================================================
print("\n--- PASSO 3: Salvando arquivos de treino e validação ---")

# Salvar Treino
if train_data: # Só salva se não estiver vazio
    try:
        with open(caminho_jsonl_treino, 'w', encoding='utf-8') as f_train:
            for item in train_data:
                f_train.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Dataset de treino salvo com sucesso em: {caminho_jsonl_treino}")
    except Exception as e:
        print(f"Erro ao salvar o arquivo de treino: {e}")
else:
    print("Aviso: Conjunto de treino está vazio. Nenhum arquivo de treino foi salvo.")


# Salvar Validação
if val_data: # Só salva se não estiver vazio
    try:
        with open(caminho_jsonl_valid, 'w', encoding='utf-8') as f_valid:
            for item in val_data:
                f_valid.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Dataset de validação salvo com sucesso em: {caminho_jsonl_valid}")
    except Exception as e:
        print(f"Erro ao salvar o arquivo de validação: {e}")
else:
    print("Aviso: Conjunto de validação está vazio. Nenhum arquivo de validação foi salvo.")


print("\n--- Divisão Concluída ---")