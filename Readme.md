# Projeto TupiAntigo2LLM: Criação de Dataset a partir de Dicionário OCRizado

## 1. Objetivo

Este projeto visa extrair, limpar e estruturar dados de um dicionário Tupi Antigo-Português digitalizado (via OCR), com o objetivo final de possibilitar o fine-tuning de Modelos de Linguagem Grandes (LLMs) para tarefas relacionadas ao Tupi Antigo (como definição de vocabulário, tradução básica, ou compreensão da estrutura da língua).

## 2. Fonte dos Dados

O material base principal é uma versão PDF OCRizada do:

*   **Dicionário Tupi Antigo-Português** de Carvalho, Moacyr Ribeiro de. (1987). Salvador: BDA.

Localizado em: `Files/txt/Carvalho_1987_DicTupiAntigo-Port_OCR.pdf`

Arquivos de texto derivados ou fontes secundárias podem estar presentes em `Files/txt/`.
```
## 3. Estrutura de Pastas
TUPI2LLM/
├── .venv/ # Ambiente virtual Python (ignorado pelo Git)
├── Files/
│ └── txt/ # Arquivos de texto fonte
│ ├── Carvalho_1987_DicTupiAntigo-Port_OCR.pdf # PDF Original OCRizado
│ ├── Carvalho_1987_DicTupiAntigo-Port_OCR.txt # Texto bruto extraído do PDF (provavelmente)
│ └── Dicionário Tupi Antigo A Língua Indígena Clássica Do Brasil.txt # Outro arquivo texto (?)
├── output/ # Arquivos gerados pelo processamento
│ ├── alfabeto/ # Processamento da seção do alfabeto/gramática
│ │ ├── alfabeto_bruto.txt # Texto bruto da seção inicial
│ │ └── alfabeto_tratato.txt # Texto limpo da seção inicial
│ ├── Arquivos_treinamento/ # Datasets formatados para uso em LLMs
│ │ ├── alfabeto_tratado.jsonl # Dataset de instrução (Q&A) sobre regras
│ │ └── dicionario_estruturado_final.json # Saída principal: dicionário em JSON estruturado
│ └── Brutos/ # Arquivos de log ou erros de versões anteriores
│ ├── entradas_com_erro_v7.txt # Log de erros v7
│ └── entradas_com_erro_v9.txt # Log de erros v9 (ou outra versão)
├── Python/
│ └── Notebooks/ # Código fonte do processamento
│ ├── .ipynb_checkpoints/ # Checkpoints automáticos do Jupyter
│ ├── DoclingParsing-checkpoint.ipynb # Notebook de trabalho (ou checkpoint)
│ └── BookParsing.ipynb # Notebook principal (provavelmente) para parsing
├── .gitattributes # Configuração do Git
└── Readme.md # Este arquivo
```
*(Nota: Os nomes exatos e o propósito de alguns arquivos como `DoclingParsing` ou o segundo `.txt` em `Files/` podem precisar de confirmação.)*

## 4. Metodologia e Workflow

O processamento foi realizado de forma iterativa, utilizando Python (principalmente nos notebooks em `Python/Notebooks/`), focando em superar os desafios do OCR e da estrutura semi-regular do dicionário:

1.  **Extração Inicial:** Leitura do texto do PDF original (`...PDF`) para um formato de texto bruto (`...txt`).
2.  **Pré-filtragem:** Remoção programática de ruídos óbvios como números de página, separadores (`*****`), cabeçalhos/rodapés identificáveis e linhas vazias. Um snapshot desse estágio pode estar em `output/Brutos/arquivo_pre_filtrado_debug.txt` (se gerado).
3.  **Divisão em Entradas:** Uso de expressões regulares (`re.split` com lookahead) para segmentar o texto filtrado em blocos correspondentes a cada entrada do dicionário. A precisão desta etapa foi refinada ao longo de várias versões.
4.  **Limpeza Robusta:** Aplicação de uma função de limpeza (`limpar_texto_robusto`) contendo um conjunto extenso de regras de substituição (`replace`) e regex (`re`) para:
    *   Corrigir erros sistemáticos de OCR (ex: `9`->`ç`, `Prefi. xo`->`Prefixo`, `nao`->`não`).
    *   Normalizar acentuação e caracteres especiais (ex: `ã`, `û`, `y`).
    *   Padronizar espaçamento e pontuação.
    *   Padronizar a formatação dos nomes das classes gramaticais (ex: `Substantivo.` -> `Substantivo:`).
    *   ***Nota:*** *Esta função requer constante atualização à medida que novos erros são identificados.*
5.  **Extração Estruturada (Iterativa):** Desenvolvimento de lógica Python com regex para analisar cada entrada limpa e extrair os seguintes campos:
    *   `verbete_tupi`: O termo principal em Tupi Antigo. *Desafio chave foi isolá-lo do texto adjacente.*
    *   `marcadores`: Indicadores entre parênteses/chaves, como `(XE)`, `(T-)`, `(Q-)`.
    *   `classe_gramatical`: A classificação da palavra (Substantivo, Verbo, etc.).
    *   `definicoes`: O(s) texto(s) explicativo(s) em português.
    *   `exemplos`: Linhas ou trechos marcados com `EX:`.
    *   `referencias`: Linhas ou trechos marcados com `=`.
6.  **Armazenamento Estruturado:** Os dados extraídos foram compilados em uma lista de dicionários Python e salvos no arquivo JSON principal: `output/Arquivos_treinamento/dicionario_estruturado_final.json`. Logs de erros de execuções anteriores foram mantidos em `output/Brutos/`.
7.  **Processamento da Seção "Alfabeto":** A seção introdutória sobre o alfabeto e regras gramaticais foi:
    *   Isolada (`output/alfabeto/alfabeto_bruto.txt`).
    *   Limpa (`output/alfabeto/alfabeto_tratato.txt`).
    *   Convertida para um formato de instrução JSONL (`output/Arquivos_treinamento/alfabeto_tratado.jsonl`) para ensinar essas regras ao LLM.
8.  **Geração de Datasets para LLM (Próximo Passo / Implícito):** O `dicionario_estruturado_final.json` e o `alfabeto_tratado.jsonl` são as bases para criar os formatos finais para fine-tuning, como:
    *   Datasets de instrução combinados (definições + regras).
    *   Datasets paralelos Tupi-Português (verbete-definição, ou de exemplos se extraídos).
    *   Corpus de texto Tupi puro.

## 5. Desafios Principais

*   **Ruído do OCR:** A maior dificuldade foi lidar com a quantidade e variedade de erros de reconhecimento de caracteres e formatação do documento original.
*   **Estrutura Semi-Regular:** Variações na formatação do dicionário tornaram desafiadora a criação de regras de extração 100% precisas.
*   **Separação Verbete/Classe/Definição:** Isolar corretamente o verbete Tupi, especialmente quando informações adjacentes apareciam na mesma linha, exigiu múltiplas iterações e refinamentos da lógica de parsing.

## 6. Ferramentas Utilizadas

*   **Linguagem:** Python 3
*   **Ambiente:** Jupyter Notebook (`BookParsing.ipynb`, etc.), possivelmente VS Code.
*   **Bibliotecas Principais:**
    *   `re` (Expressões Regulares)
    *   `json`
    *   `csv`
    *   `os`
    *   `pandas` (Para análise e validação)
    *   `PyPDF2` / `PyMuPDF` (Para extração inicial do PDF)
    *   `sklearn` (Para divisão treino/validação)

## 7. Artefatos Gerados (Principais)

*   `output/Arquivos_treinamento/dicionario_estruturado_final.json`: **Dataset principal.** Dicionário completo em formato JSON estruturado.
*   `output/Arquivos_treinamento/alfabeto_tratado.jsonl`: Dataset de instrução sobre regras do alfabeto/gramática.
*   `(Gerados posteriormente a partir do JSON)`:
    *   `dataset_tupi_vX_final.txt`: Lista de verbetes Tupi.
    *   `dataset_paralelo_vX_final.csv`: Pares Tupi (verbete) - Português (definição).
    *   `dataset_instrucao_vX_final.jsonl`: Dataset para fine-tuning de instrução (definições).
    *   `train_dataset.jsonl` / `validation_dataset.jsonl`: Divisões para treino/validação.
*   `output/Brutos/entradas_com_erro_vX.txt`: Logs de erros de execuções anteriores.

## 8. Próximos Passos

1.  **Validação e Refinamento Final:** Revisão manual e/ou programática do `dicionario_estruturado_final.json` para corrigir erros remanescentes. Idealmente, refinar o(s) notebook(s) de extração em `Python/Notebooks/` se forem encontrados problemas sistemáticos.
2.  **Geração dos Datasets de Treinamento Finais:** Criar os arquivos `.txt`, `.csv` ou `.jsonl` definitivos a partir do JSON validado, aplicando a limpeza final durante a geração.
3.  **Fine-tuning do LLM:**
    *   Selecionar um modelo base pré-treinado.
    *   Configurar o ambiente de treinamento (GPU, bibliotecas).
    *   Executar o fine-tuning usando os datasets divididos (treino/validação) e técnicas como LoRA/QLoRA.
4.  **Avaliação:** Testar o desempenho do modelo fine-tuned nas tarefas desejadas.
5.  **Iteração:** Refinar os datasets ou o processo de fine-tuning com base na avaliação.