# private-legal-assistant

Sistema que implementa um assistente virtual para orientação jurídica sobre o registro civil garantindo a privacidade do usuário.

## Pré-requisitos

Antes de executar o sistema, é necessário instalar algumas dependências.

### 1. Instalar o Poetry

O **Poetry** é utilizado para gerenciamento de dependências nesse projeto. Para instalá-lo, execute o seguinte comando:  
`curl -sSL https://install.python-poetry.org | python3 -l`

### 2. Instalar as dependências do projeto
Com o **Poetry** instalado, Instale as dependencias com:  
`poetry install`

### 3. Instalar o Ollama
O Ollama é utilizado para rodar modelos de IA para embeddings e processamento de queries. Para instalá-lo, utilize o comando abaixo:  
`curl -fsSL https://ollama.com/install.sh | sh`

### 4. Instalar o Ollama
Em um terminal separado, inicie o ollama server através do comando:  
`ollama serve`

### 5. Baixar os modelos necessários
De modo a garantir que o assistente tenha acesso aos modelos corretos, baixe os modelos abaixo:  
`ollama pull nomic-embed-text`
`ollama pull llama3.2:3b-instruct-fp16`

<!-- Em um outro terminal rode o modelo:  
`ollama run llama3.2:3b-instruct-fp16 --keepalive -1m` -->

### 6. Executar o programa
Para rodar o programa, execute o seguinte comando:  
`poetry run python3 legal-assistant`
