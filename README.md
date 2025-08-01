# agent_rag

Implementação de um projeto com [LangChain](https://www.langchain.com/), combinando RAG (Retrieval-Augmented Generation) com agentes inteligentes.

> 🔍 **Baseado nos repositórios de:**
>
> * [Thomas Janssen](https://github.com/ThomasJanssen-tech/Retrieval-Augmented-Generation/tree/main)
> * [Alejandro AO](https://github.com/alejandro-ao/ask-multiple-pdfs)


---

## ✅ Pré-requisitos

* Python **3.10+**
* Git instalado
* Conta e chave de API da [OpenAI](https://platform.openai.com/account/api-keys)

---

## ⚙️ Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/igormelo-sd/agent_rag.git
```

### 2. Acesse o diretório

```bash
cd langchain_agent_rag\agent_rag\rag
```

### 3. Crie um ambiente virtual

```bash
python -m venv venv
```
Use o interpretador python na versão 3.10 

### 4. Ative o ambiente virtual


**Windows (CMD):**

```bash
venv\Scripts\activate
```

**PowerShell:**

```bash
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
source venv/bin/activate
```
### 5. Acesse o diretório novamente

```bash
cd agent_rag\langchain
```

### 6. Instale as dependências

```bash
pip install -r requirements.txt
```

### 7. Configure sua chave de API da OpenAI

1. Obtenha a chave em: [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)
2. Crie um arquivo `.env` na mesma pasta 
3. Adicione sua chave no campo apropriado(`OPENAI_API_KEY= "sk-..."`)

---

## ▶️ Executando o projeto

### Na pasta `data` troque o PDF do arquivo por um PDF de seu desejo

Para guardar o PDF no banco vetorial precisa ir até o arquivo embedding.py, colocar o nome desejado da coleção na linha 216 e na linha 375 e no arquivo rag_system.py na linha 21, o nome não pode possuir caracteres especiais e então rodar o arquivo
```bash
python populate_database.py
```

### Rodar o agente via terminal

```bash
python agent.py
```

###  OU

### Rodar via Streamlit (interface web)

```bash
streamlit run streamlit_app.py
```
###  OU

```bash
python main.py
```

* Sempre que for abrir o programa use `cd langchain_agent_rag` para não dá erro no programa

---

## 🧠 Tecnologias utilizadas

* [LangChain](https://www.langchain.com/)
* [OpenAI API](https://platform.openai.com/)
* [Streamlit](https://streamlit.io/)
* Python 3.13+

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](./LICENSE) para mais detalhes.
