# rag_system_fixed.py - Sistema RAG Otimizado e Corrigido
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import List, Dict, Any, Optional, Tuple
import logging
import csv
from datetime import datetime
import numpy as np
import warnings

# Suprimir warnings desnecessários
warnings.filterwarnings("ignore", category=UserWarning)

# Configuração de logging otimizada
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Importação condicional do reranker com fallback
try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
    logger.info("✅ sentence-transformers disponível")
except ImportError:
    RERANKER_AVAILABLE = False
    logger.warning("⚠️ sentence-transformers não disponível. Reranking desabilitado.")

class RagSystem:
    """Sistema RAG robusto com inicialização melhorada e tratamento de erros avançado."""
    
    def __init__(self, 
                 chroma_path: str = "chroma_db", 
                 collection_name: str = "seade_gecon",
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 enable_reranking: bool = True,
                 enable_logging: bool = True,
                 max_retries: int = 3,
                 **kwargs):
        """
        Inicializa o sistema RAG com verificações robustas.
        
        Args:
            chroma_path: Caminho para o banco ChromaDB
            collection_name: Nome da coleção
            reranker_model: Modelo para reranqueamento
            enable_reranking: Habilitar reranqueamento
            enable_logging: Habilitar logging
            max_retries: Máximo de tentativas para operações
        """
        # Carregar variáveis de ambiente
        load_dotenv()
        
        # Configurações do sistema
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.enable_reranking = enable_reranking and RERANKER_AVAILABLE
        self.enable_logging = enable_logging
        self.max_retries = max_retries
        self.is_initialized = False
        
        # Inicialização dos componentes
        self.chroma_client = None
        self.collection = None
        self.openai_client = None
        self.reranker = None
        
        # Desabilitar telemetria do ChromaDB
        os.environ["ANONYMIZED_TELEMETRY"] = "False"
        
        # Executar inicialização
        self._initialize_system()
    
    def _initialize_system(self) -> None:
        """Inicialização robusta do sistema com verificações detalhadas."""
        logger.info("🚀 Iniciando sistema RAG...")
        
        # Etapa 1: Verificar variáveis de ambiente
        if not self._check_environment():
            logger.error("❌ Falha na verificação do ambiente")
            return
        
        # Etapa 2: Inicializar ChromaDB
        if not self._init_chroma_safe():
            logger.error("❌ Falha na inicialização do ChromaDB")
            return
        
        # Etapa 3: Inicializar OpenAI
        if not self._init_openai_safe():
            logger.error("❌ Falha na inicialização do OpenAI")
            return
        
        # Etapa 4: Inicializar reranker (opcional)
        if self.enable_reranking:
            self._init_reranker_safe()
        
        # Etapa 5: Configurar logging
        if self.enable_logging:
            self._init_logging_safe()
        
        # Verificação final
        self.is_initialized = self._verify_system_ready()
        
        if self.is_initialized:
            logger.info("✅ Sistema RAG inicializado com sucesso")
            self._log_system_status()
        else:
            logger.error("❌ Sistema RAG não pôde ser completamente inicializado")
    
    def _check_environment(self) -> bool:
        """Verifica se o ambiente está configurado corretamente."""
        try:
            # Verificar OPENAI_API_KEY
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.error("OPENAI_API_KEY não encontrada no ambiente")
                logger.error("Soluções:")
                logger.error("1. Criar arquivo .env: OPENAI_API_KEY=sk-seu-token-aqui")
                logger.error("2. Definir variável: export OPENAI_API_KEY=sk-seu-token-aqui")
                return False
            
            if not api_key.startswith('sk-'):
                logger.error("OPENAI_API_KEY parece inválida (deve começar com 'sk-')")
                return False
            
            logger.info(f"✅ OPENAI_API_KEY encontrada: {api_key[:10]}...")
            
            # Verificar se o diretório ChromaDB pode ser criado
            os.makedirs(self.chroma_path, exist_ok=True)
            logger.info(f"✅ Diretório ChromaDB: {self.chroma_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro na verificação do ambiente: {e}")
            return False
    
    def _init_chroma_safe(self) -> bool:
        """Inicialização segura do ChromaDB com múltiplas tentativas."""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Tentativa {attempt + 1}/{self.max_retries} - Inicializando ChromaDB...")
                
                # Criar cliente persistente
                self.chroma_client = chromadb.PersistentClient(
                    path=self.chroma_path,
                    settings=chromadb.config.Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
                
                # Tentar obter ou criar coleção
                try:
                    self.collection = self.chroma_client.get_collection(name=self.collection_name)
                    doc_count = self.collection.count()
                    logger.info(f"✅ Coleção existente encontrada: {self.collection_name} ({doc_count} docs)")
                    
                    if doc_count == 0:
                        logger.warning("⚠️ Coleção existe mas está vazia!")
                        logger.warning("Execute o processo de indexação para popular a base")
                    
                except chromadb.errors.InvalidCollectionException:
                    logger.info(f"Coleção {self.collection_name} não existe, criando...")
                    self.collection = self.chroma_client.create_collection(
                        name=self.collection_name,
                        metadata={"description": "Dados econômicos de São Paulo"}
                    )
                    logger.info(f"✅ Nova coleção criada: {self.collection_name}")
                
                # Teste de funcionalidade básica
                if self._test_chroma_basic():
                    return True
                else:
                    logger.warning("Teste básico do ChromaDB falhou")
                    
            except Exception as e:
                logger.error(f"Tentativa {attempt + 1} falhou: {e}")
                if attempt == self.max_retries - 1:
                    logger.error("❌ Todas as tentativas de inicialização do ChromaDB falharam")
                    logger.error("Soluções:")
                    logger.error("1. pip install chromadb --upgrade")
                    logger.error("2. Remover pasta chroma_db e tentar novamente")
                    logger.error("3. Verificar permissões de escrita no diretório")
                    return False
        
        return False
    
    def _test_chroma_basic(self) -> bool:
        """Teste básico de funcionalidade do ChromaDB."""
        try:
            # Teste simples de operação
            test_result = self.collection.count()
            logger.info(f"✅ Teste ChromaDB: {test_result} documentos na coleção")
            return True
        except Exception as e:
            logger.error(f"Falha no teste básico do ChromaDB: {e}")
            return False
    
    def _init_openai_safe(self) -> bool:
        """Inicialização segura do cliente OpenAI."""
        try:
            self.openai_client = OpenAI()
            
            # Teste básico da API
            try:
                # Fazer uma requisição simples para testar a conectividade
                models = self.openai_client.models.list()
                logger.info("✅ Cliente OpenAI inicializado e testado")
                return True
            except Exception as api_error:
                logger.error(f"Erro no teste da API OpenAI: {api_error}")
                logger.error("Verifique se a OPENAI_API_KEY é válida e tem créditos")
                return False
                
        except Exception as e:
            logger.error(f"Erro ao inicializar cliente OpenAI: {e}")
            return False
    
    def _init_reranker_safe(self) -> None:
        """Inicialização segura do reranker."""
        if not RERANKER_AVAILABLE:
            logger.info("Reranker não disponível - continuando sem reranking")
            self.enable_reranking = False
            return
        
        try:
            logger.info("Carregando modelo de reranking...")
            self.reranker = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                max_length=512
            )
            logger.info("✅ Reranker inicializado com sucesso")
        except Exception as e:
            logger.warning(f"Falha ao inicializar reranker: {e}")
            logger.info("Continuando sem reranking...")
            self.enable_reranking = False
    
    def _init_logging_safe(self) -> None:
        """Inicialização segura do sistema de logging."""
        try:
            self.log_file = "rag_queries.csv"
            
            # Criar arquivo de log se não existir
            if not os.path.exists(self.log_file):
                with open(self.log_file, "w", encoding="utf-8", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp", "query", "response_length", 
                        "num_documents", "confidence_avg", "processing_time_ms"
                    ])
            
            logger.info(f"✅ Sistema de logging ativo: {self.log_file}")
            
        except Exception as e:
            logger.warning(f"Falha ao inicializar logging: {e}")
            self.enable_logging = False
    
    def _verify_system_ready(self) -> bool:
        """Verifica se o sistema está completamente pronto."""
        checks = {
            "chroma_client": self.chroma_client is not None,
            "collection": self.collection is not None,
            "openai_client": self.openai_client is not None
        }
        
        all_ready = all(checks.values())
        
        if all_ready:
            logger.info("✅ Todos os componentes prontos")
        else:
            logger.error("❌ Componentes com problemas:")
            for component, status in checks.items():
                if not status:
                    logger.error(f"  - {component}: FALHOU")
        
        return all_ready
    
    def _log_system_status(self) -> None:
        """Log do status completo do sistema."""
        try:
            doc_count = self.collection.count() if self.collection else 0
            
            logger.info("📊 STATUS DO SISTEMA RAG:")
            logger.info(f"  - ChromaDB: ✅ Conectado")
            logger.info(f"  - Coleção: {self.collection_name}")
            logger.info(f"  - Documentos: {doc_count}")
            logger.info(f"  - OpenAI: ✅ Conectado")
            logger.info(f"  - Reranking: {'✅ Ativo' if self.enable_reranking else '❌ Inativo'}")
            logger.info(f"  - Logging: {'✅ Ativo' if self.enable_logging else '❌ Inativo'}")
            
        except Exception as e:
            logger.warning(f"Erro ao obter status: {e}")
    
    def retrieve_documents(self, query: str, n_results: int = 8) -> Tuple[List[str], List[float]]:
        """Recupera documentos com tratamento robusto de erros."""
        if not self.is_initialized or not self.collection:
            logger.error("Sistema não inicializado para busca")
            return [], []
        
        try:
            # Verificar se há documentos na coleção
            doc_count = self.collection.count()
            if doc_count == 0:
                logger.warning("Coleção vazia - nenhuma busca possível")
                return [], []
            
            # Ajustar n_results para não exceder documentos disponíveis
            actual_n_results = min(n_results, doc_count)
            
            # Executar busca
            results = self.collection.query(
                query_texts=[query],
                n_results=actual_n_results,
                include=['documents', 'distances', 'metadatas']
            )
            
            documents = []
            distances = []
            
            if results and results.get('documents') and results['documents'][0]:
                documents = results['documents'][0]
                raw_distances = results.get('distances', [[]])[0]
                
                # Converter distâncias em scores de confiança (invertido e normalizado)
                if raw_distances:
                    # Normalizar distâncias para scores de 0 a 1
                    max_dist = max(raw_distances) if raw_distances else 1.0
                    distances = [1.0 - (d / max_dist) for d in raw_distances]
                else:
                    distances = [0.5] * len(documents)
                
                logger.info(f"✅ Recuperados {len(documents)} documentos")
            
            return documents, distances
            
        except Exception as e:
            logger.error(f"Erro na busca de documentos: {e}")
            return [], []
    
    def rerank_documents(self, query: str, documents: List[str], top_k: int = 4) -> Tuple[List[str], List[float]]:
        """Reranking com fallback robusto."""
        if not documents:
            return [], []
        
        # Se reranking não está disponível, retornar os primeiros documentos
        if not self.enable_reranking or not self.reranker:
            selected_docs = documents[:top_k]
            # Scores decrescentes simulados
            scores = [0.8 - (i * 0.1) for i in range(len(selected_docs))]
            return selected_docs, scores
        
        try:
            # Criar pares para o reranker
            pairs = [[query, doc] for doc in documents]
            
            # Calcular scores
            scores = self.reranker.predict(pairs)
            
            # Garantir que scores é uma lista
            if hasattr(scores, 'tolist'):
                scores = scores.tolist()
            elif not isinstance(scores, (list, tuple)):
                scores = [float(scores)]
            
            # Converter para float e ordenar
            doc_score_pairs = list(zip(documents, [float(s) for s in scores]))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Retornar top_k
            ranked_docs = [doc for doc, _ in doc_score_pairs[:top_k]]
            confidence_scores = [score for _, score in doc_score_pairs[:top_k]]
            
            logger.info(f"✅ Reranking concluído - Top score: {max(confidence_scores):.3f}")
            return ranked_docs, confidence_scores
            
        except Exception as e:
            logger.error(f"Erro no reranking: {e}")
            # Fallback: retornar documentos originais
            selected_docs = documents[:top_k]
            fallback_scores = [0.6 - (i * 0.1) for i in range(len(selected_docs))]
            return selected_docs, fallback_scores
    
    def generate_response(self, query: str, documents: List[str], confidence_scores: List[float]) -> str:
        """Gera resposta usando OpenAI com prompt otimizado."""
        try:
            if not self.openai_client:
                return "❌ Cliente OpenAI não disponível"
            
            # Preparar contexto dos documentos
            if documents:
                doc_context = []
                for i, (doc, score) in enumerate(zip(documents, confidence_scores)):
                    score_safe = float(score) if score is not None else 0.0
                    doc_context.append(f"**Documento {i+1}** (Relevância: {score_safe:.2f})\n{doc}")
                
                context_text = "\n\n".join(doc_context)
                avg_confidence = np.mean([s for s in confidence_scores if s is not None])
            else:
                context_text = "⚠️ Nenhum documento relevante encontrado na base de dados."
                avg_confidence = 0.0
            
            # Prompt otimizado
            system_prompt = f"""Você é um especialista em economia do Estado de São Paulo.

BASE SUA RESPOSTA EXCLUSIVAMENTE nos documentos fornecidos abaixo.

INSTRUÇÕES:
- Use apenas informações dos documentos fornecidos
- Se não houver informações suficientes, diga isso claramente
- Estruture a resposta de forma clara e profissional
- Inclua dados específicos e números quando disponíveis
- Use português formal e técnico

DOCUMENTOS RELEVANTES:
{context_text}

CONFIANÇA MÉDIA DOS DADOS: {avg_confidence:.2f}

Responda à pergunta com base exclusivamente nos documentos acima."""

            # Fazer requisição para OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.3,
                max_tokens=4000,
                top_p=0.9
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Erro ao gerar resposta: {e}")
            return f"❌ Erro ao processar sua consulta: {str(e)}"
    
    def query(self, user_query: str, n_results: int = 6) -> Dict[str, Any]:
        """
        Método principal de consulta - COMPATÍVEL COM agent.py
        
        Args:
            user_query: Pergunta do usuário
            n_results: Número de resultados desejados
            
        Returns:
            Dicionário com resposta e metadados
        """
        start_time = datetime.now()
        
        # Verificar se sistema está pronto
        if not self.is_initialized:
            return {
                "query": user_query,
                "response": "❌ Sistema RAG não inicializado. Verifique a configuração.",
                "error": "Sistema não inicializado",
                "num_documents": 0,
                "processing_time_ms": 0,
                "confidence_scores": [],
                "quality_assessment": {"quality_score": 0.0, "has_sufficient_data": False}
            }
        
        try:
            logger.info(f"🔍 Processando consulta: {user_query}")
            
            # Buscar documentos
            documents, confidence_scores = self.retrieve_documents(user_query, n_results * 2)
            
            # Reranking se disponível
            if documents:
                documents, confidence_scores = self.rerank_documents(
                    user_query, documents, top_k=min(n_results, len(documents))
                )
            
            # Gerar resposta
            response = self.generate_response(user_query, documents, confidence_scores)
            
            # Calcular métricas
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
            
            # Assessment de qualidade
            quality_assessment = {
                "quality_score": float(avg_confidence),
                "has_sufficient_data": len(documents) > 0 and avg_confidence > 0.2,
                "recommendation": "good" if avg_confidence > 0.5 else "moderate" if avg_confidence > 0.2 else "low"
            }
            
            # Log da consulta
            if self.enable_logging:
                self._log_query_safe(user_query, response, len(documents), avg_confidence, processing_time)
            
            return {
                "query": user_query,
                "response": response,
                "retrieved_documents": documents,
                "confidence_scores": confidence_scores,
                "num_documents": len(documents),
                "quality_assessment": quality_assessment,
                "processing_time_ms": processing_time,
                "reranking_enabled": self.enable_reranking,
                "system_initialized": self.is_initialized
            }
            
        except Exception as e:
            logger.error(f"Erro durante consulta: {e}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                "query": user_query,
                "response": f"❌ Erro interno do sistema: {str(e)}",
                "error": str(e),
                "num_documents": 0,
                "processing_time_ms": processing_time,
                "confidence_scores": [],
                "quality_assessment": {"quality_score": 0.0, "has_sufficient_data": False}
            }
    
    def _log_query_safe(self, query: str, response: str, num_docs: int, confidence_avg: float, processing_time_ms: float) -> None:
        """Logging seguro de consultas."""
        if not self.enable_logging:
            return
        
        try:
            with open(self.log_file, "a", encoding="utf-8", newline='') as f:
                writer = csv.writer(f)
                # Limpar quebras de linha na query
                clean_query = query.replace('\n', ' ').replace('\r', '')[:200]
                writer.writerow([
                    datetime.now().isoformat(),
                    clean_query,
                    len(response),
                    num_docs,
                    f"{confidence_avg:.3f}",
                    f"{processing_time_ms:.1f}"
                ])
        except Exception as e:
            logger.warning(f"Erro no logging: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Retorna status detalhado do sistema."""
        status = {
            "initialized": self.is_initialized,
            "chroma_client": self.chroma_client is not None,
            "openai_client": self.openai_client is not None,
            "collection_exists": self.collection is not None,
            "collection_count": 0,
            "reranking_enabled": self.enable_reranking,
            "logging_enabled": self.enable_logging,
            "chroma_path": self.chroma_path,
            "collection_name": self.collection_name
        }
        
        try:
            if self.collection:
                status["collection_count"] = self.collection.count()
        except Exception as e:
            status["collection_error"] = str(e)
        
        return status
    
    def reset_system(self) -> bool:
        """Reinicializa o sistema em caso de problemas."""
        logger.info("🔄 Reinicializando sistema RAG...")
        
        # Limpar componentes
        self.chroma_client = None
        self.collection = None
        self.openai_client = None
        self.reranker = None
        self.is_initialized = False
        
        # Reinicializar
        self._initialize_system()
        
        return self.is_initialized
    
    def test_system(self) -> Dict[str, Any]:
        """Teste completo do sistema."""
        test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "overall_status": "unknown",
            "recommendations": []
        }
        
        # Teste 1: Inicialização
        test_results["tests"]["initialization"] = {
            "status": "passed" if self.is_initialized else "failed",
            "details": "Sistema inicializado" if self.is_initialized else "Sistema não inicializado"
        }
        
        # Teste 2: ChromaDB
        try:
            if self.collection:
                doc_count = self.collection.count()
                test_results["tests"]["chromadb"] = {
                    "status": "passed",
                    "details": f"{doc_count} documentos na coleção"
                }
                if doc_count == 0:
                    test_results["recommendations"].append("Popular base de dados com documentos")
            else:
                test_results["tests"]["chromadb"] = {
                    "status": "failed",
                    "details": "Coleção não disponível"
                }
        except Exception as e:
            test_results["tests"]["chromadb"] = {
                "status": "failed",
                "details": str(e)
            }
        
        # Teste 3: OpenAI
        try:
            if self.openai_client:
                # Teste simples
                test_response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=10
                )
                test_results["tests"]["openai"] = {
                    "status": "passed",
                    "details": "API OpenAI respondendo"
                }
            else:
                test_results["tests"]["openai"] = {
                    "status": "failed",
                    "details": "Cliente OpenAI não inicializado"
                }
        except Exception as e:
            test_results["tests"]["openai"] = {
                "status": "failed",
                "details": str(e)
            }
        
        # Teste 4: Consulta completa (se possível)
        if self.is_initialized:
            try:
                result = self.query("teste sistema")
                if "error" not in result:
                    test_results["tests"]["full_query"] = {
                        "status": "passed",
                        "details": f"Consulta processada em {result['processing_time_ms']:.1f}ms"
                    }
                else:
                    test_results["tests"]["full_query"] = {
                        "status": "failed",
                        "details": result.get("error", "Erro desconhecido")
                    }
            except Exception as e:
                test_results["tests"]["full_query"] = {
                    "status": "failed",
                    "details": str(e)
                }
        
        # Avaliação geral
        passed_tests = sum(1 for test in test_results["tests"].values() if test["status"] == "passed")
        total_tests = len(test_results["tests"])
        
        if passed_tests == total_tests:
            test_results["overall_status"] = "excellent"
        elif passed_tests >= total_tests * 0.75:
            test_results["overall_status"] = "good"
        elif passed_tests >= total_tests * 0.5:
            test_results["overall_status"] = "needs_attention"
        else:
            test_results["overall_status"] = "critical"
        
        return test_results

# Função para criar instância do sistema
def create_rag_system(**kwargs) -> RagSystem:
    """Cria uma instância do sistema RAG com configurações padrão otimizadas."""
    return RagSystem(
        enable_reranking=True,
        enable_logging=True,
        max_retries=3,
        **kwargs
    )

# Teste básico se executado diretamente
if __name__ == "__main__":
    print("🚀 Testando Sistema RAG...")
    
    try:
        rag = create_rag_system()
        
        # Executar teste do sistema
        test_results = rag.test_system()
        
        print(f"\n📊 Resultados dos testes:")
        print(f"Status geral: {test_results['overall_status']}")
        
        for test_name, test_info in test_results["tests"].items():
            status_icon = "✅" if test_info["status"] == "passed" else "❌"
            print(f"{status_icon} {test_name}: {test_info['details']}")
        
        if test_results["recommendations"]:
            print(f"\n💡 Recomendações:")
            for rec in test_results["recommendations"]:
                print(f"  - {rec}")
        
        # Teste de consulta simples se sistema estiver ok
        if rag.is_initialized:
            print(f"\n🔍 Teste de consulta simples...")
            result = rag.query("economia São Paulo")
            print(f"Resposta: {result['response'][:200]}...")
            print(f"Documentos: {result['num_documents']}")
            print(f"Tempo: {result['processing_time_ms']:.1f}ms")
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        print(f"\n🛠️ Soluções:")
        print(f"1. pip install chromadb sentence-transformers openai python-dotenv")
        print(f"2. Criar arquivo .env com OPENAI_API_KEY")
        print(f"3. Verificar se há documentos na base ChromaDB")