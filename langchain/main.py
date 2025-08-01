import subprocess
import os

# Caminho do arquivo Streamlit (ajuste se estiver em outra pasta)
streamlit_file = "streamlit_app.py"

# Verifica se o arquivo existe antes de tentar rodar
if not os.path.exists(streamlit_file):
    print(f"Erro: O arquivo '{streamlit_file}' não foi encontrado.")
else:
    print(f"Iniciando Streamlit com '{streamlit_file}'...")
    try:
        # Executa o comando como se fosse: streamlit run streamlit_app.py
        subprocess.run(["streamlit", "run", streamlit_file])
    except Exception as e:
        print(f"Erro ao executar Streamlit: {e}")
