# populate_database.py - Popular Base de Dados RAG
import os
import sys
from pathlib import Path
import logging
from typing import List, Dict, Any
import hashlib

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Tentar importar dependências
try:
    from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import chromadb
    from chromadb.utils import embedding_functions
    print("✅ Todas as dependências importadas com sucesso!")
except ImportError as e:
    print(f"❌ Erro ao importar dependências: {e}")
    print("Execute: pip install langchain-community chromadb")
    sys.exit(1)

def check_pdf_files(data_path: str = "data") -> List[str]:
    """Verifica e lista arquivos PDF disponíveis."""
    print(f"🔍 Verificando arquivos PDF em: {data_path}")
    
    data_dir = Path(data_path)
    if not data_dir.exists():
        print(f"❌ Diretório não encontrado: {data_path}")
        return []
    
    pdf_files = list(data_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ Nenhum PDF encontrado em {data_path}")
        return []
    
    print(f"📄 Encontrados {len(pdf_files)} arquivo(s) PDF:")
    for pdf in pdf_files:
        size_mb = pdf.stat().st_size / (1024 * 1024)
        print(f"  - {pdf.name} ({size_mb:.1f} MB)")
    
    return [str(pdf) for pdf in pdf_files]

def process_documents_to_chromadb(data_path: str = "data", 
                                chroma_path: str = "chroma_db",
                                collection_name: str = "seade_gecon"):
    """Processa documentos PDF e adiciona ao ChromaDB."""
    
    print(f"\n🚀 PROCESSANDO DOCUMENTOS PARA CHROMADB")
    print(f"=" * 50)
    print(f"📁 Origem: {data_path}")
    print(f"🗄️ Destino: {chroma_path}")
    print(f"📚 Coleção: {collection_name}")
    
    # Verificar PDFs
    pdf_files = check_pdf_files(data_path)
    if not pdf_files:
        print("❌ Nenhum PDF para processar")
        return False
    
    try:
        # Configurar ChromaDB
        print(f"\n🔧 Configurando ChromaDB...")
        os.makedirs(chroma_path, exist_ok=True)
        
        chroma_client = chromadb.PersistentClient(
            path=chroma_path,
            settings=chromadb.config.Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Configurar função de embedding
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Obter ou criar coleção
        try:
            collection = chroma_client.get_collection(name=collection_name)
            print(f"✅ Coleção existente encontrada: {collection_name}")
            
            # Verificar se já tem documentos
            current_count = collection.count()
            if current_count > 0:
                print(f"📊 Coleção já contém {current_count} documentos")
                
                overwrite = input("❓ Deseja sobrescrever os documentos existentes? (s/N): ").strip().lower()
                if overwrite in ['s', 'sim', 'y', 'yes']:
                    print("🗑️ Removendo coleção existente...")
                    chroma_client.delete_collection(name=collection_name)
                    collection = chroma_client.create_collection(
                        name=collection_name,
                        embedding_function=embedding_function
                    )
                    print("✅ Nova coleção criada")
                else:
                    print("❌ Processamento cancelado pelo usuário")
                    return False
                    
        except Exception:
            # Coleção não existe, criar nova
            collection = chroma_client.create_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            print(f"✅ Nova coleção criada: {collection_name}")
        
        # Carregar documentos
        print(f"\n📖 Carregando documentos PDF...")
        loader = PyPDFDirectoryLoader(data_path)
        documents = loader.load()
        
        if not documents:
            print("❌ Nenhum documento carregado")
            return False
        
        print(f"✅ {len(documents)} páginas carregadas")
        
        # Configurar text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        
        # Dividir em chunks
        print(f"✂️ Dividindo em chunks...")
        chunks = text_splitter.split_documents(documents)
        print(f"✅ {len(chunks)} chunks criados")
        
        # Filtrar chunks válidos (mais de 50 caracteres)
        valid_chunks = [chunk for chunk in chunks if len(chunk.page_content.strip()) > 50]
        print(f"✅ {len(valid_chunks)} chunks válidos")
        
        if not valid_chunks:
            print("❌ Nenhum chunk válido para processar")
            return False
        
        # Preparar dados para inserção
        print(f"🔄 Preparando dados para inserção...")
        documents_list = []
        metadata_list = []
        ids_list = []
        
        for i, chunk in enumerate(valid_chunks):
            # Criar ID único baseado no conteúdo
            content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:8]
            doc_id = f"doc_{i}_{content_hash}"
            
            documents_list.append(chunk.page_content)
            metadata_list.append(chunk.metadata)
            ids_list.append(doc_id)
        
        # Inserir em lotes
        print(f"💾 Inserindo documentos no ChromaDB...")
        batch_size = 100
        total_batches = (len(documents_list) + batch_size - 1) // batch_size
        
        for i in range(0, len(documents_list), batch_size):
            batch_docs = documents_list[i:i+batch_size]
            batch_metadata = metadata_list[i:i+batch_size]
            batch_ids = ids_list[i:i+batch_size]
            
            try:
                collection.upsert(
                    documents=batch_docs,
                    metadatas=batch_metadata,
                    ids=batch_ids
                )
                
                current_batch = (i // batch_size) + 1
                print(f"  📦 Lote {current_batch}/{total_batches} inserido ({len(batch_docs)} docs)")
                
            except Exception as e:
                print(f"❌ Erro no lote {current_batch}: {e}")
                continue
        
        # Verificar resultado final
        final_count = collection.count()
        print(f"\n🎉 PROCESSAMENTO CONCLUÍDO!")
        print(f"📊 Total de documentos na coleção: {final_count}")
        
        # Teste de consulta
        if final_count > 0:
            print(f"\n🔍 Testando consulta...")
            try:
                test_results = collection.query(
                    query_texts=["economia"],
                    n_results=3
                )
                
                if test_results and test_results.get('documents'):
                    print(f"✅ Teste de consulta bem-sucedido!")
                    print(f"📄 Primeira resposta: {test_results['documents'][0][0][:100]}...")
                else:
                    print("⚠️ Consulta retornou resultado vazio")
                    
            except Exception as e:
                print(f"⚠️ Erro no teste de consulta: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro durante o processamento: {e}")
        logger.error(f"Erro no processamento: {e}")
        return False

def main():
    """Função principal."""
    print("🚀 POPULAR BASE DE DADOS RAG")
    print("=" * 50)
    
    # Verificar se diretório data existe
    if not os.path.exists("data"):
        print("❌ Diretório 'data' não encontrado")
        print("💡 Crie o diretório 'data' e coloque seus arquivos PDF nele")
        return
    
    # Processar documentos
    success = process_documents_to_chromadb(
        data_path="data",
        chroma_path="chroma_db", 
        collection_name="seade_gecon"
    )
    
    if success:
        print(f"\n✅ Base de dados populada com sucesso!")
        print(f"💡 Agora você pode usar o sistema RAG para fazer consultas")
        print(f"🚀 Execute: python rag_system_fixed.py")
    else:
        print(f"\n❌ Falha ao popular base de dados")
        print(f"🛠️ Verifique os logs acima para detalhes do erro")

if __name__ == "__main__":
    main()