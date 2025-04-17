# Tupi2LLM

Tupi2LLM é um projeto que visa processar e estruturar dados lexicográficos da língua Tupi Antigo, extraídos de documentos históricos, para criar datasets utilizáveis em modelos de linguagem natural (LLMs). O objetivo é preservar e facilitar o estudo da língua Tupi Antigo, permitindo sua aplicação em tecnologias modernas.

## Estrutura do Projeto

O projeto está organizado da seguinte forma:


### Principais Arquivos

- **`Files/`**: Contém os documentos originais, como PDFs e arquivos de texto, que servem como fonte de dados.
- **`output/`**: Diretório onde os resultados do processamento são armazenados, incluindo arquivos intermediários e finais, como datasets e dicionários estruturados.
- **`Python/Notebooks/BookParsing.ipynb`**: Notebook principal que realiza a extração, limpeza e estruturação dos dados.

## Funcionalidades

1. **Extração de Texto**:
   - Utiliza bibliotecas como `PyPDF2` ou `PyMuPDF` para extrair texto de PDFs.
   - Salva o texto bruto extraído em `output/arquivo_bruto_extraido.txt`.

2. **Pré-filtragem e Limpeza**:
   - Remove linhas irrelevantes ou com erros de OCR.
   - Normaliza o texto para corrigir erros comuns.

3. **Estruturação de Dados**:
   - Divide o texto em entradas lexicográficas.
   - Identifica e organiza informações como verbetes, definições, exemplos e referências.

4. **Geração de Datasets**:
   - Cria datasets em formatos como JSON, TXT e CSV para uso em modelos de linguagem ou análises.

## Requisitos

- Python 3.8 ou superior
- Bibliotecas Python:
  - `PyPDF2`
  - `PyMuPDF`
  - `pandas`
  - `json`
  - `re`

## Como Usar

1. **Instale as dependências**:
   ```bash
   pip install PyPDF2 fitz pandas
   Execute o notebook principal:

Abra Python/Notebooks/BookParsing.ipynb no Jupyter Notebook ou JupyterLab.
Siga as etapas descritas no notebook para processar os arquivos.
Resultados:

Os arquivos processados serão salvos no diretório output/.
Contribuição
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.

Licença
Este projeto está licenciado sob a MIT License.

Autor
Desenvolvido por Peterson Pereira. ```