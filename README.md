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
O **Ollama** é utilizado para rodar modelos de IA para embeddings e processamento de queries. Para instalá-lo, utilize o comando abaixo:  
`curl -fsSL https://ollama.com/install.sh | sh`

### 4. Baixar os modelos necessários
De modo a garantir que o assistente tenha acesso aos modelos esperados pela aplicação, baixe os modelos abaixo:  
`ollama pull nomic-embed-text`  
`ollama pull gemma:2b`  
**Obs.**: Os modelos são consideravelmente grandes; recomenda-se que haja pelo menos 10 GB de armazenamento livre no sistema.

## Executar o programa
Para rodar o programa, execute o seguinte comando:  
`poetry run python3 legal-assistant`

**Obs.**: Para atualizar a base de dados antes de iniciar o programa, utilize o comando:   
`poetry run python3 legal-assistant --update-db`