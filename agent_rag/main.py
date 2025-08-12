import subprocess
import os
import sys

# Importa a função de processamento de documentos
from embedding import process_documents_to_chromadb

# Caminho do arquivo Streamlit
streamlit_file = "streamlit_app.py"

def run_streamlit():
    """Função para iniciar a aplicação Streamlit."""
    print(f"Iniciando Streamlit com '{streamlit_file}'...")
    try:
        subprocess.run(["streamlit", "run", streamlit_file])
    except FileNotFoundError:
        print("Erro: 'streamlit' não é reconhecido como um comando.")
        print("Verifique se o Streamlit está instalado (pip install streamlit).")
    except Exception as e:
        print(f"Erro ao executar Streamlit: {e}")

def main():
    """Função principal que gerencia o fluxo do programa."""
    
    # Verifica se a base de dados precisa ser populada
    if "--populate" in sys.argv:
        print("\n🚀 Iniciando a criação da base de dados vetorial...")
        try:
            # Chama a função do seu arquivo embedding.py
            process_documents_to_chromadb(
                data_path="data", 
                chroma_path="chroma_db", 
                collection_name="seade_gecon"
            )
            print("✅ Base de dados criada/atualizada com sucesso!")
        except Exception as e:
            print(f"❌ Erro durante o processamento de embeddings: {e}")
            return # Sai se a base de dados não puder ser criada

    # Verifica se o arquivo do Streamlit existe
    if not os.path.exists(streamlit_file):
        print(f"Erro: O arquivo '{streamlit_file}' não foi encontrado.")
    else:
        run_streamlit()

if __name__ == "__main__":
    main()