# private-legal-assistant

Sistema que implementa um assistente virtual para orientação jurídica sobre o registro civil garantindo a privacidade do usuário.

Primeiramente instale o poetry com o comando: 
`curl -sSL https://install.python-poetry.org | python3 -`

Instale as dependencias com:
`poetry install`

Baixe e instale o Ollama:
`curl -fsSL https://ollama.com/install.sh | sh`

Em um terminal separado, inicie o ollama server através do comando:
`ollama serve`

Em um outro terminal rode o modelo:
`ollama run llama3.2:3b-instruct-fp16 --keepalive -1m`

Execute o programa com:
`poetry run python3 legal-assistant`
