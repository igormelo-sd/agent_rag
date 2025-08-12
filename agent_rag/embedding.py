# embedding.py - Popular Base de Dados RAG (VERSÃO CORRIGIDA)
import os
import sys
from pathlib import Path
import logging
from typing import List, Dict, Any
import base64
import io
import hashlib
import json
from dotenv import load_dotenv
import time

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

print("🔧 Verificando variáveis de ambiente...")
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ OPENAI_API_KEY não encontrada no .env!")
    print("Por favor, crie um arquivo .env na raiz do projeto com:")
    print("OPENAI_API_KEY=sk-seu-token-aqui")
    sys.exit(1)
else:
    print(f"✅ OPENAI_API_KEY encontrada: {api_key[:10]}...")

# Criar função de retry simples (sem dependência externa)
def retry_with_exponential_backoff(max_retries=3, base_delay=1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    delay = base_delay * (2 ** attempt)
                    print(f"Erro na tentativa {attempt + 1}: {e}")
                    print(f"Tentando novamente em {delay}s...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

# Tentar importar dependências.
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import chromadb
    from chromadb.utils import embedding_functions
    from openai import OpenAI
    import fitz  # PyMuPDF
    from PIL import Image
    
    print("✅ Todas as dependências importadas com sucesso!")
except ImportError as e:
    print(f"❌ Erro ao importar dependências: {e}")
    print("Por favor, execute:")
    print("pip install langchain-text-splitters chromadb openai PyMuPDF Pillow python-dotenv")
    sys.exit(1)

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuração da API OpenAI
try:
    client_openai = OpenAI(api_key=api_key)
    print("✅ Cliente OpenAI inicializado")
except Exception as e:
    print(f"❌ Erro ao inicializar cliente OpenAI: {e}")
    sys.exit(1)

# Inicialização do cache
CACHE_DIR = "cache"
Path(CACHE_DIR).mkdir(exist_ok=True)

def encode_image_to_base64(image: Image.Image) -> str:
    """Converte um objeto de imagem PIL para uma string Base64."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@retry_with_exponential_backoff(max_retries=3, base_delay=2)
def describe_image_with_openai(image_base64: str, page_hash: str) -> str:
    """
    Usa a API da OpenAI (GPT-4o) para descrever o conteúdo de uma imagem com cache.
    """
    cache_path = Path(CACHE_DIR) / f"{page_hash}.json"
    
    if cache_path.exists():
        logger.info(f"  -> Usando cache para a página {page_hash[:8]}...")
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)["description"]
        except Exception as e:
            logger.warning(f"Erro ao ler cache, gerando nova descrição: {e}")

    try:
        logger.info(f"  -> Gerando descrição visual com OpenAI...")
        response = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Descreva detalhadamente o conteúdo visual desta imagem, "
                                "incluindo todos os gráficos, tabelas, diagramas e texto presente. "
                                "Traduza o texto para português se necessário e resuma as informações-chave. "
                                "A descrição deve ser o mais completa e objetiva possível para ser usada em um sistema de busca."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=2048,
        )
        description = response.choices[0].message.content

        # Salva a descrição no cache
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"description": description}, f, ensure_ascii=False)
            logger.info(f"  -> Descrição salva no cache")
        except Exception as e:
            logger.warning(f"Erro ao salvar cache: {e}")
            
        return description
    except Exception as e:
        logger.error(f"Erro ao descrever a imagem com OpenAI: {e}")
        return "Não foi possível gerar uma descrição para esta imagem."


def test_pdf_access(pdf_path: Path) -> bool:
    """Testa se o PDF pode ser aberto e processado."""
    try:
        logger.info(f"🧪 Testando acesso ao PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        page_count = doc.page_count
        logger.info(f"✅ PDF acessível: {page_count} páginas")
        
        # Testar primeira página
        if page_count > 0:
            page = doc[0]
            text = page.get_text()
            logger.info(f"✅ Texto da primeira página extraído: {len(text)} caracteres")
            
            # Testar conversão de imagem
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            logger.info(f"✅ Imagem extraída: {img.width}x{img.height}")
            
        doc.close()
        return True
    except Exception as e:
        logger.error(f"❌ Erro ao testar PDF: {e}")
        return False

def load_and_process_multimodal_documents(data_path: str = "data") -> List[Dict[str, Any]]:
    """
    Carrega PDFs, extrai texto e imagens, e gera descrições multimodais.
    """
    logger.info(f"🔍 Verificando arquivos PDF em: {os.path.abspath(data_path)}")
    data_dir = Path(data_path)
    
    if not data_dir.exists():
        logger.error(f"❌ Diretório não encontrado: {data_path}")
        logger.info(f"Criando diretório: {data_path}")
        data_dir.mkdir(parents=True, exist_ok=True)
        return []

    pdf_files = list(data_dir.glob("*.pdf"))
    logger.info(f"📁 Arquivos PDF encontrados: {len(pdf_files)}")
    for pdf in pdf_files:
        logger.info(f"  - {pdf.name} ({pdf.stat().st_size / 1024:.1f} KB)")
    
    if not pdf_files:
        logger.error(f"❌ Nenhum PDF encontrado em {data_path}")
        return []

    documents = []
    total_pages = 0
    
    for pdf_path in pdf_files:
        logger.info(f"\n📄 Processando o arquivo PDF: {pdf_path.name}")
        
        # Testar acesso ao PDF primeiro
        if not test_pdf_access(pdf_path):
            logger.error(f"❌ Pulando arquivo {pdf_path.name} devido a erro de acesso")
            continue
            
        try:
            doc = fitz.open(pdf_path)
            logger.info(f"  -> PDF aberto com sucesso: {doc.page_count} páginas")
            
            for i, page in enumerate(doc):
                page_num = i + 1
                logger.info(f"  -> Processando página {page_num}/{doc.page_count}...")
                
                try:
                    # Extrair texto
                    text_content = page.get_text()
                    logger.info(f"     Texto extraído: {len(text_content)} caracteres")
                    
                    # Gerar hash da página
                    page_data = f"{pdf_path.name}-page-{page_num}-{text_content[:100]}".encode('utf-8')
                    page_hash = hashlib.sha256(page_data).hexdigest()
                    
                    # Extrair imagem da página
                    logger.info(f"     Convertendo página para imagem...")
                    pix = page.get_pixmap(dpi=200) 
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                    # Redimensionar se necessário
                    max_size = 1024
                    if img.width > max_size or img.height > max_size:
                        logger.info(f"     Redimensionando imagem de {img.width}x{img.height}")
                        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    
                    # Converter para base64
                    img_base64 = encode_image_to_base64(img)
                    logger.info(f"     Imagem convertida para base64: {len(img_base64)} caracteres")
                    
                    # Gerar descrição da imagem
                    img_description = describe_image_with_openai(img_base64, page_hash)
                    logger.info(f"     Descrição gerada: {len(img_description)} caracteres")

                    # Combinar conteúdos
                    full_content = (
                        f"--- Conteúdo da Página {page_num} do arquivo '{pdf_path.name}' ---\n\n"
                        f"TEXTO DA PÁGINA:\n{text_content}\n\n"
                        f"DESCRIÇÃO VISUAL (Gráficos, Imagens, etc.):\n{img_description}\n"
                    )
                    
                    documents.append({
                        "content": full_content,
                        "metadata": {
                            "source": str(pdf_path),
                            "page": page_num,
                            "file_name": pdf_path.name,
                            "content_length": len(full_content),
                            "text_length": len(text_content),
                            "description_length": len(img_description)
                        }
                    })
                    
                    total_pages += 1
                    logger.info(f"     ✅ Página {page_num} processada com sucesso")
                    
                except Exception as e:
                    logger.error(f"     ❌ Erro na página {page_num}: {e}")
                    continue
                    
            doc.close()
            logger.info(f"✅ Arquivo {pdf_path.name} processado completamente")
            
        except Exception as e:
            logger.error(f"❌ Erro ao processar o arquivo {pdf_path.name}: {e}")
            continue
    
    logger.info(f"\n✅ RESUMO: {len(documents)} páginas processadas de {total_pages} páginas totais")
    return documents

def test_chromadb_connection(chroma_path: str, collection_name: str) -> bool:
    """Testa a conexão com o ChromaDB."""
    try:
        logger.info(f"🧪 Testando conexão ChromaDB: {chroma_path}")
        
        # Verificar se o diretório existe
        if not Path(chroma_path).exists():
            logger.info(f"Criando diretório ChromaDB: {chroma_path}")
            Path(chroma_path).mkdir(parents=True, exist_ok=True)
        
        chroma_client = chromadb.PersistentClient(path=chroma_path)
        
        # Testar função de embedding
        ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
        
        # Testar criação de coleção
        collection = chroma_client.get_or_create_collection(
            name=f"{collection_name}_test", 
            embedding_function=ef
        )
        
        # Testar adição de documento
        collection.add(
            documents=["Teste de conexão"],
            metadatas=[{"test": True}],
            ids=["test-id"]
        )
        
        # Testar consulta
        results = collection.query(query_texts=["teste"], n_results=1)
        
        # Limpar teste
        chroma_client.delete_collection(f"{collection_name}_test")
        
        logger.info("✅ ChromaDB funcionando corretamente")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao testar ChromaDB: {e}")
        return False

def process_documents_to_chromadb(data_path: str = "data", chroma_path: str = "chroma_db", collection_name: str = "seade_gecon"):
    """
    Processa documentos PDF multimodais e adiciona ao ChromaDB.
    """
    print(f"\n🚀 INICIANDO PROCESSAMENTO DE DOCUMENTOS")
    print(f"📁 Diretório de dados: {os.path.abspath(data_path)}")
    print(f"🗄️ ChromaDB: {os.path.abspath(chroma_path)}")
    print(f"📚 Coleção: {collection_name}")
    
    # Testar ChromaDB primeiro
    if not test_chromadb_connection(chroma_path, collection_name):
        logger.error("❌ Falha na conexão com ChromaDB. Abortando.")
        return
    
    # Processar documentos
    documents_raw = load_and_process_multimodal_documents(data_path)
    if not documents_raw:
        logger.warning("❌ Nenhum documento processado. Finalizando.")
        return

    logger.info(f"\n📝 Dividindo documentos em chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=500,
        separators=["\n\n", "\n", " ", ""]
    )
    
    docs_to_embed = []
    chunk_count = 0
    
    for doc_idx, doc in enumerate(documents_raw):
        logger.info(f"  -> Processando documento {doc_idx + 1}/{len(documents_raw)}")
        try:
            chunks = text_splitter.split_text(doc['content'])
            logger.info(f"     Gerados {len(chunks)} chunks")
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"{doc['metadata']['file_name']}-page{doc['metadata']['page']}-chunk{chunk_idx}"
                
                docs_to_embed.append({
                    "document": chunk,
                    "metadata": {
                        **doc['metadata'],
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_idx,
                        "total_chunks": len(chunks)
                    },
                    "id": hashlib.sha256(chunk_id.encode()).hexdigest()
                })
                chunk_count += 1
                
        except Exception as e:
            logger.error(f"     ❌ Erro ao dividir documento: {e}")

    logger.info(f"✅ Total de {len(docs_to_embed)} chunks prontos para embedding.")

    try:
        logger.info(f"\n🗄️ Conectando ao ChromaDB...")
        chroma_client = chromadb.PersistentClient(path=chroma_path)
        
        ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
        
        # Remover coleção existente se existir
        try:
            chroma_client.delete_collection(collection_name)
            logger.info(f"  -> Coleção existente '{collection_name}' removida")
        except:
            pass
        
        collection = chroma_client.create_collection(
            name=collection_name, 
            embedding_function=ef
        )
        logger.info(f"  -> Coleção '{collection_name}' criada")
        
        # Adicionar documentos em lotes
        batch_size = 50
        total_batches = (len(docs_to_embed) + batch_size - 1) // batch_size
        
        logger.info(f"📦 Adicionando {len(docs_to_embed)} documentos em {total_batches} lotes...")
        
        for i in range(0, len(docs_to_embed), batch_size):
            batch_num = (i // batch_size) + 1
            batch_docs = docs_to_embed[i:i + batch_size]
            
            logger.info(f"  -> Processando lote {batch_num}/{total_batches} ({len(batch_docs)} documentos)...")
            
            try:
                texts = [d['document'] for d in batch_docs]
                metadatas = [d['metadata'] for d in batch_docs]
                ids = [d['id'] for d in batch_docs]
                
                collection.add(
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"     ✅ Lote {batch_num} adicionado com sucesso")
                
            except Exception as e:
                logger.error(f"     ❌ Erro no lote {batch_num}: {e}")
                continue
        
        # Verificar resultado final
        final_count = collection.count()
        logger.info(f"\n🎉 PROCESSAMENTO CONCLUÍDO!")
        logger.info(f"   📊 Total de documentos na coleção: {final_count}")
        
        # Teste de consulta
        logger.info(f"🧪 Testando consulta...")
        test_results = collection.query(
            query_texts=["economia"], 
            n_results=1
        )
        
        if test_results['documents']:
            logger.info(f"   ✅ Consulta teste funcionou: {len(test_results['documents'][0])} resultado(s)")
        else:
            logger.warning(f"   ⚠️ Consulta teste não retornou resultados")
        
    except Exception as e:
        logger.error(f"❌ Erro ao processar ChromaDB: {e}")
        raise

def main():
    """Função principal para executar o processamento."""
    print("🚀 SISTEMA DE EMBEDDING MULTIMODAL")
    print("=" * 50)
    
    # Verificar estrutura de diretórios
    data_path = "data"
    chroma_path = "chroma_db"
    
    print(f"📁 Verificando estrutura de diretórios...")
    print(f"   Data: {os.path.abspath(data_path)} {'✅' if Path(data_path).exists() else '❌'}")
    print(f"   ChromaDB: {os.path.abspath(chroma_path)} {'✅' if Path(chroma_path).exists() else '⚠️ (será criado)'}")
    
    if not Path(data_path).exists():
        logger.error(f"❌ Diretório {data_path} não existe!")
        logger.info("Crie o diretório e adicione seus arquivos PDF.")
        return
    
    # Executar processamento
    try:
        process_documents_to_chromadb(
            data_path=data_path,
            chroma_path=chroma_path,
            collection_name="seade_gecon"
        )
        print(f"\n🎉 PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Processamento interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ ERRO DURANTE O PROCESSAMENTO: {e}")
        logger.exception("Detalhes do erro:")

if __name__ == "__main__":
    main()