# Projeto TupiAntigo2LLM: Fine-tuning de LLM com Dicionário Tupi Antigo

**(Status Atual: Fine-tuning inicial concluído, avaliação em andamento)**

![](https://miro.medium.com/v2/resize:fit:720/format:webp/1*wXtFuUNYL6Zr07efqa-X_A.png)

## 1. Objetivo

Este projeto visa criar um dataset estruturado a partir de um dicionário Tupi Antigo-Português digitalizado (via OCR) e utilizá-lo para realizar o fine-tuning de um Modelo de Linguagem Grande (LLM). O objetivo é desenvolver um modelo capaz de compreender e realizar tarefas básicas relacionadas ao Tupi Antigo, como fornecer definições de vocabulário e, potencialmente, auxiliar na preservação digital e estudo dessa língua ancestral brasileira.

## 2. Fonte dos Dados

O material base principal é uma versão PDF OCRizada do:

*   **Dicionário Tupi Antigo-Português** de Carvalho, Moacyr Ribeiro de. (1987). Salvador: BDA.

Localizado em: `Files/txt/Carvalho_1987_DicTupiAntigo-Port_OCR.pdf`

## 3. Estrutura de Pastas (Atualizada)
```
TUPI2LLM/
├── .venv/ # Ambiente virtual Python
├── Files/
│ └── txt/ # Arquivos de texto fonte (PDF, TXT bruto)
│ ├── Carvalho_1987_DicTupiAntigo-Port_OCR.pdf
│ └── Carvalho_1987_DicTupiAntigo-Port_OCR.txt
├── output/ # Arquivos gerados pelo processamento
│ ├── alfabeto/ # Processamento da seção do alfabeto/gramática
│ │ ├── alfabeto_bruto.txt
│ │ └── alfabeto_tratato.txt
│ ├── Arquivos_treinamento/ # Datasets formatados para LLM e logs
│ │ ├── alfabeto_tratado.jsonl # Instruções de Gramática
│ │ ├── dicionario_estruturado_final_vX.json # JSON estruturado (usar versão final, ex: v8)
│ │ ├── dataset_instrucao_vX_final.jsonl # Dataset principal de instruções (definições)
│ │ ├── train_dataset.jsonl # Dataset de Treino (divisão do JSONL de instrução)
│ │ └── validation_dataset.jsonl # Dataset de Validação (divisão do JSONL de instrução)
│ │ └── dataset_tupi_vX_final.txt # (Opcional) Dataset de texto Tupi puro
│ │ └── dataset_paralelo_vX_final.csv # (Opcional) Dataset CSV Tupi-Definição
│ └── Brutos/ # Arquivos de log ou erros
│ └── entradas_com_erro_vX.txt # Logs de erro da extração/pós-processamento
│ └── arquivo_pre_filtrado_debug.txt # Texto após pré-filtragem (opcional)
├── Python/
│ └── Notebooks/ # Código fonte do processamento e testes
│ ├── BookParsing.ipynb # Notebook para parsing do dicionário (provavelmente)
│ ├── split_dataset.py # Script para dividir o JSONL em treino/validação
│ └── inference_test.py # Script/Notebook para testar o modelo fine-tuned
├── tupi-gemma-2b-lora-v8/ # Diretório de SAÍDA do Axolotl com adaptadores LoRA <<< AJUSTE O NOME
│ ├── adapter_config.json
│ ├── adapter_model.safetensors
│ └── ... (outros arquivos salvos pelo Axolotl/Trainer)
├── .gitattributes
├── config.yaml # Arquivo de configuração do Axolotl para fine-tuning
└── Readme.md # Este arquivo
```
*(Nota: Substitua `vX` pela versão final utilizada nos nomes dos arquivos. Ajuste o nome do diretório de saída do Axolotl)*

## 4. Metodologia e Workflow Executado

1.  **Extração Inicial e Pré-filtragem:** Texto extraído do PDF e limpeza inicial para remover ruídos grosseiros.
2.  **Limpeza Robusta:** Uso intensivo de Python e Regex para corrigir erros de OCR e normalizar o texto (etapa crucial e contínua).
3.  **Divisão e Extração Estrutural:** Segmentação do texto em entradas e extração iterativa de campos (`verbete_tupi`, `classe_gramatical`, `definicoes`, etc.) para um arquivo JSON (`dicionario_estruturado_final_vX.json`).
4.  **Processamento da Gramática:** Seção inicial do dicionário limpa e formatada em `alfabeto_tratado.jsonl`.
5.  **Pós-processamento (Limpeza Final):** Aplicação de limpeza adicional nos dados extraídos do JSON (ex: v8) para refinar verbetes e definições antes de gerar os datasets de treinamento.
6.  **Geração do Dataset de Instrução:** Conversão dos dados limpos (verbete-definição) para o formato JSON Lines (`dataset_instrucao_vX_final.jsonl`), adequado para fine-tuning de instrução.
7.  **Divisão Treino/Validação:** O dataset `.jsonl` foi dividido (~90%/10%) usando `split_dataset.py` para criar `train_dataset.jsonl` e `validation_dataset.jsonl`.
8.  **Configuração do Ambiente de Treinamento:**
    *   Utilização do **WSL 2 (Windows Subsystem for Linux)** no Windows 11 para melhor compatibilidade das bibliotecas de ML.
    *   Criação de um ambiente **Conda** (`tupi_llm`) com Python 3.10.
    *   Instalação das bibliotecas necessárias: `pytorch` (com suporte a CUDA), `transformers`, `datasets`, `accelerate`, `peft`, `trl`, `bitsandbytes`, `sentencepiece`.
    *   Instalação do `build-essential` no WSL para permitir a compilação do `bitsandbytes`.
    *   Autenticação no Hugging Face Hub (`huggingface-cli login`) e aceite dos termos do modelo base.
9.  **Fine-tuning com Axolotl:**
    *   **Modelo Base:** `google/gemma-2b-it` (escolhido por ser pequeno e bom em instruções).
    *   **Técnica:** **QLoRA** (LoRA com quantização de 4 bits) para viabilizar o treinamento na GPU GTX 1650 (4GB VRAM).
    *   **Configuração (`config.yaml`):** Ajustes específicos para baixa VRAM, incluindo `sequence_len: 256`, `load_in_4bit: true`, `micro_batch_size: 1`, `gradient_accumulation_steps: 8`, `lora_r: 8`, `gradient_checkpointing: true`.
    *   **Execução:** Treinamento iniciado usando `accelerate launch -m axolotl.cli.train config.yaml`.
    *   **Resultado:** Adaptadores LoRA salvos em `./tupi-gemma-2b-lora-v8` (ou nome similar).
10. **Configuração do Ambiente de Inferência (Colab):**
    *   Criação de um notebook no Google Colab com acesso a GPU (T4).
    *   Instalação das mesmas dependências (`pip install ...`).
    *   Login no Hugging Face Hub.
11. **Teste/Avaliação Inicial:**
    *   Carregamento do modelo base (`gemma-2b-it`) com quantização QLoRA.
    *   Aplicação dos adaptadores LoRA treinados (baixados do Hub ou localmente).
    *   Execução de um script (`inference_test.py`) para gerar definições para palavras Tupi e avaliar qualitativamente as respostas.

## 5. Desafios Enfrentados (Atualizado)

*   **Qualidade do OCR:** Continua sendo um desafio de base, exigindo limpeza contínua.
*   **Estrutura Variável do Dicionário:** Dificultou a extração 100% precisa.
*   **Isolamento de Campos:** Separar `verbete_tupi` da classe/definição foi o maior desafio da extração.
*   **Limitações de Hardware (VRAM):** A GTX 1650 exigiu o uso de QLoRA, `sequence_len` baixo (256), `micro_batch_size` 1 e `gradient_checkpointing` para conseguir rodar o fine-tuning do Gemma 2B.
*   **Configuração do Ambiente (WSL):** Necessidade de instalar `build-essential` para `bitsandbytes` e configurar corretamente o DNS para acesso ao Hugging Face Hub.
*   **Modelos Gated (Hugging Face):** Necessidade de aceitar termos de uso e autenticar (`huggingface-cli login`).
*   **Carregamento de Adaptadores PEFT:** Encontro de `KeyError` devido a possíveis problemas com `device_map` e quantização, necessitando ajustes na forma de carregar modelo + adaptadores para inferência.

## 6. Ferramentas Utilizadas (Atualizado)

*   **Linguagem:** Python 3
*   **Ambiente:** Jupyter Notebook, WSL 2 (Ubuntu), Google Colab
*   **Bibliotecas Principais:** `re`, `json`, `csv`, `os`, `pandas`, `PyPDF2`/`PyMuPDF`, `sklearn`, `torch`, `transformers`, `datasets`, `accelerate`, `peft`, `bitsandbytes`, `trl`, `sentencepiece`
*   **Framework de Fine-tuning:** `axolotl`

## 7. Artefatos Gerados (Principais)

*   `output/Arquivos_treinamento/dicionario_estruturado_final_vX.json`: JSON estruturado (versão final usada como base).
*   `output/Arquivos_treinamento/alfabeto_tratado.jsonl`: Instruções de gramática.
*   `output/Arquivos_treinamento/train_dataset.jsonl`: Dataset de treino para LLM.
*   `output/Arquivos_treinamento/validation_dataset.jsonl`: Dataset de validação para LLM.
*   `tupi-gemma-2b-lora-v8/` (ou similar): **Diretório contendo os adaptadores LoRA resultantes do fine-tuning.**
*   `Python/Notebooks/inference_test.py`: Script/Notebook para carregar e testar o modelo fine-tuned.
*   `config.yaml`: Arquivo de configuração do Axolotl usado para o treinamento.

## 8. Próximos Passos Imediatos

<<<<<<< HEAD
1.  **Avaliação Qualitativa Detalhada:** Usar o `inference_test.py` para testar uma gama maior de palavras (incluindo as que falharam na extração ou estavam no set de validação) e analisar a qualidade, coerência e precisão das definições geradas.
2.  **Análise de Erros:** Identificar os tipos de erros mais comuns que o modelo comete. Ele alucina? Confunde palavras? Gera respostas genéricas? Falha com termos específicos?
3.  **Decisão Baseada na Avaliação:**
    *   **Se Suficiente para Testes:** Prosseguir com experimentações adicionais ou integração em uma aplicação simples.
    *   **Se Necessita Melhoria (Provável):**
        *   **Prioridade 1: Melhorar Dados:** Voltar aos scripts de limpeza (`limpar_texto_robusto`) e extração (Passo 5 da v8) para gerar um JSON de entrada *melhor* e *retreinar*. Corrigir os problemas remanescentes no isolamento do verbete e limpeza das definições é crucial.
        *   **Prioridade 2: Ajustar Treinamento:** Experimentar com mais épocas (`num_epochs: 2` ou `3` no `config.yaml`), ajustar a taxa de aprendizado, ou o rank do LoRA (`lora_r`).
        *   **Prioridade 3: Mais Dados:** Incorporar o `alfabeto_tratado.jsonl` ao treinamento. Buscar outras fontes textuais.

## 9. Como Usar os Artefatos Atuais

*   **`dicionario_estruturado_final_vX.json`:** Fonte para análise e geração de novos formatos de dataset.
*   **`train/validation_dataset.jsonl`:** Usados diretamente para retreinar/continuar o fine-tuning com Axolotl ou TRL.
*   **`tupi-gemma-2b-lora-v8/` (Adaptadores):** Carregar junto com o modelo base `google/gemma-2b-it` para realizar inferência, usando o `inference_test.py` como exemplo. Compartilhar esta pasta (ou subir para o Hugging Face Hub) permite que outros usem seu modelo fine-tuned.
*   **`config.yaml`:** Documenta os parâmetros usados e pode ser modificado para novos treinamentos.

![Google Colab](https://colab.research.google.com/drive/147oT9B2e-FNg4KGFMFkqhqu-CBYcJaFK?usp=sharing)
# 📘 Tupi2LLM - Inferência com Modelo Quantizado e LoRA

Este notebook demonstra como carregar um modelo base de linguagem grande (LLM) com **quantização 4-bit (QLoRA)** e aplicar adaptadores **LoRA personalizados** para inferência eficiente em GPUs com memória limitada (como T4).

## 🔧 Requisitos
- GPU com suporte a `bfloat16` (ex: T4, A100)
- Conta no [Hugging Face](https://huggingface.co) com token configurado
- Modelos salvos no Hub com arquivos necessários:
  - `config.json` com `model_type`
  - pesos adaptadores (LoRA)

## 📦 Dependências
- `transformers`
- `peft`
- `bitsandbytes`
- `torch`

As bibliotecas serão instaladas automaticamente no início do notebook.

## 📑 Etapas do Notebook
1. Instalação das bibliotecas
2. Carregamento do tokenizer
3. Configuração de quantização (QLoRA)
4. Carregamento do modelo base quantizado
5. Aplicação dos adaptadores LoRA
6. Inferência pronta! 🎯

## 📁 Identificadores dos Modelos
- **BASE_MODEL_ID**: `meta-llama/Llama-2-7b-hf` (ou equivalente)
- **ADAPTER_HUB_ID**: `peterson047/Tupi2LLM`

> 💡 Dica: Os adaptadores devem ser compatíveis com o modelo base!
=======
1.  **Validação e Refinamento Final:** Revisão manual e/ou programática do `dicionario_estruturado_final.json` para corrigir erros remanescentes. Idealmente, refinar o(s) notebook(s) de extração em `Python/Notebooks/` se forem encontrados problemas sistemáticos.
2.  **Geração dos Datasets de Treinamento Finais:** Criar os arquivos `.txt`, `.csv` ou `.jsonl` definitivos a partir do JSON validado, aplicando a limpeza final durante a geração.
3.  **Fine-tuning do LLM:**
    *   Selecionar um modelo base pré-treinado.
    *   Configurar o ambiente de treinamento (GPU, bibliotecas).
    *   Executar o fine-tuning usando os datasets divididos (treino/validação) e técnicas como LoRA/QLoRA.
4.  **Avaliação:** Testar o desempenho do modelo fine-tuned nas tarefas desejadas.
5.  **Iteração:** Refinar os datasets ou o processo de fine-tuning com base na avaliação.
>>>>>>> 1baf3ff5714f6e835e004bf46f1b93b60ae19ee6
