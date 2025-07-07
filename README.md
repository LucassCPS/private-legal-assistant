<div align="center">
  <h1>Private Legal Assistant</h1>
  <p>
    <strong>Um assistente virtual focado em privacidade para orientação jurídica sobre registro civil.</strong>
  </p>
  <p>
    <a href="#-sobre-o-projeto">Sobre</a> •
    <a href="#-configuração">Configuração</a> •
    <a href="#-como-executar">Execução</a>
  </p>
</div>

## Sobre o Projeto

Este sistema implementa um assistente jurídico virtual projetado para fornecer orientações sobre questões de registro civil e focado em preservar a privacidade do usuário.

## Configuração

Antes de executar o sistema, é necessário instalar algumas dependências.

### 1. Instalar o Poetry

O **Poetry** é utilizado para gerenciamento de dependências nesse projeto. Para instalá-lo, siga o [manual](https://python-poetry.org/docs/#ci-recommendations).

Verifique se o **Poetry** está acessível:   
`poetry --version`

### 2. Instalar as dependências do projeto
Com o **Poetry** instalado, Instale as dependencias com:  
`poetry install`

### 3. Instalar o Ollama
O **Ollama** é utilizado para rodar modelos de IA para embeddings e processamento de queries. Para instalá-lo, utilize o comando abaixo:  
`curl -fsSL https://ollama.com/install.sh | sh`

### 4. Baixar os modelos necessários
De modo a garantir que o assistente tenha acesso aos modelos esperados pela aplicação, baixe os modelos abaixo:  
`ollama pull nomic-embed-text`   
`ollama pull mistral:7b`  
`ollama pull gemma3:1b`  

**Obs.**: Os modelos são grandes. Recomenda-se ter pelo menos 6 GB de espaço livre em disco (nomic-embed-text: ~275MB, mistral:7b: ~4.1GB, gemma:2b: ~1.7GB).

## Como Executar

### 1. Interface Web (recomendado)
Para rodar o programa com interface web, execute o comando:   
`poetry run streamlit run legal_assistant/app.py --server.fileWatcherType none`

Acesse o endereço fornecido no terminal (geralmente http://localhost:8501) em seu navegador.

### 2. Terminal
Para rodar o programa e utilizá-lo via terminal, execute o seguinte comando:  
`poetry run legal-assistant`

**Obs.**: Para atualizar a base de dados antes de iniciar o programa, utilize o comando:   
`poetry run legal-assistant --update-db`