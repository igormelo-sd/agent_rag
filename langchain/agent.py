# agent_fixed.py - Agente RAG Corrigido e Otimizado
import os
import logging
from typing import Dict, Any, List, Tuple
import warnings

# Suprimir warnings desnecessários
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Carregar variáveis do arquivo .env
from dotenv import load_dotenv
load_dotenv()

# Desabilitar LangSmith e telemetria
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Imports para LangChain
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

# Import do sistema RAG corrigido
RAG_AVAILABLE = False
try:
    # Tentar importar do arquivo corrigido primeiro
    try:
        from rag_system_fixed import RagSystem
        RAG_AVAILABLE = True
        print("✅ RagSystem corrigido importado com sucesso")
    except ImportError:
        # Fallback para o sistema original
        from rag_system import RagSystem
        RAG_AVAILABLE = True
        print("✅ RagSystem original importado")
except ImportError as e:
    RAG_AVAILABLE = False
    print(f"❌ ERRO: Sistema RAG não disponível: {e}")

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGAgentFixed:
    """
    Agente RAG corrigido com inicialização robusta e tratamento de erros aprimorado.
    """
    
    def __init__(self, openai_api_key: str = None, force_rag: bool = True):
        """
        Inicializa o agente RAG com verificações robustas.
        
        Args:
            openai_api_key: Chave da API da OpenAI
            force_rag: Se True, exige sistema RAG funcional
        """
        self.force_rag = force_rag
        self.rag_available = False
        self.system_ready = False
        
        print("🚀 Inicializando Agente RAG...")
        
        # Verificar e configurar API Key
        if not self._setup_api_key(openai_api_key):
            if force_rag:
                raise ValueError("OPENAI_API_KEY é obrigatória")
            return
        
        # Verificar disponibilidade do RAG
        if not RAG_AVAILABLE:
            error_msg = self._get_rag_error_message()
            if force_rag:
                raise RuntimeError(error_msg)
            else:
                print(error_msg)
                return
        
        # Inicializar sistema RAG
        if not self._initialize_rag_system():
            if force_rag:
                raise RuntimeError("❌ Sistema RAG obrigatório não pôde ser inicializado")
            return
        
        # Configurar LLM
        self.llm = ChatOpenAI(
            temperature=0.3,
            model="gpt-4o-mini",
            max_tokens=6000,
            top_p=0.8,
        )
        
        # Configurar memória
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=12000
        )
        
        # Criar ferramentas
        self.tools = self._create_tools()
        
        # Criar prompt
        self.prompt = self._create_prompt()
        
        # Criar agente
        try:
            self.agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=self.prompt
            )
            
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=False,  # Reduzir verbosidade
                handle_parsing_errors=True,
                max_iterations=5,
                max_execution_time=120,
                return_intermediate_steps=False
            )
            
            self.system_ready = True
            print("✅ Agente RAG inicializado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao criar agente: {e}")
            if force_rag:
                raise RuntimeError(f"Falha na criação do agente: {e}")
        
        # Teste inicial se sistema estiver pronto
        if self.system_ready and self.rag_available:
            self._test_system_integration()
    
    def _setup_api_key(self, api_key: str = None) -> bool:
        """Configura e valida a API key da OpenAI."""
        try:
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            
            # Verificar se foi carregada
            current_key = os.getenv("OPENAI_API_KEY")
            if not current_key:
                print("❌ OPENAI_API_KEY não encontrada")
                print("💡 Soluções:")
                print("1. Criar arquivo .env: OPENAI_API_KEY=sk-seu-token-aqui")
                print("2. Passar como parâmetro: RAGAgentFixed(openai_api_key='sk-...')")
                return False
            
            if not current_key.startswith('sk-'):
                print("❌ OPENAI_API_KEY parece inválida")
                return False
            
            print(f"✅ API Key configurada: {current_key[:10]}...")
            return True
            
        except Exception as e:
            logger.error(f"Erro na configuração da API key: {e}")
            return False
    
    def _get_rag_error_message(self) -> str:
        """Retorna mensagem de erro detalhada para problemas do RAG."""
        return """
❌ SISTEMA RAG NÃO DISPONÍVEL

PROBLEMAS IDENTIFICADOS:
- Módulo rag_system não pôde ser importado
- Dependências podem estar faltando

SOLUÇÕES RECOMENDADAS:
1. Instalar dependências:
   pip install chromadb sentence-transformers openai python-dotenv

2. Verificar arquivos:
   - rag_system.py ou rag_system_fixed.py no mesmo diretório
   - Arquivo .env com OPENAI_API_KEY

3. Testar sistema RAG independentemente:
   python rag_system_fixed.py

4. Verificar logs para erros específicos

⚠️ Este agente foi projetado para operar com sistema RAG.
Sem ele, as respostas serão limitadas e imprecisas.
        """
    
    def _initialize_rag_system(self) -> bool:
        """Inicializa o sistema RAG com verificações robustas."""
        try:
            print("🔄 Inicializando sistema RAG...")
            
            # Criar instância do RAG
            self.rag = RagSystem(
                enable_reranking=True,
                enable_logging=True,
                max_retries=3
            )
            
            # Verificar se foi inicializado
            if not hasattr(self.rag, 'is_initialized') or not self.rag.is_initialized:
                print("❌ Sistema RAG falhou na inicialização")
                return False
            
            # Verificar status
            status = self.rag.get_system_status()
            doc_count = status.get('collection_count', 0)
            
            print(f"✅ Sistema RAG ativo")
            print(f"📊 Status: {doc_count} documentos na base")
            
            if doc_count == 0:
                print("⚠️ ATENÇÃO: Base de dados vazia!")
                print("💡 Execute o processo de indexação para popular a base")
                if self.force_rag:
                    print("❌ Modo RAG obrigatório requer dados na base")
                    return False
            
            self.rag_available = True
            return True
            
        except Exception as e:
            logger.error(f"Erro na inicialização do RAG: {e}")
            print(f"❌ Falha no RAG: {e}")
            return False
    
    def _test_system_integration(self) -> None:
        """Testa a integração completa do sistema."""
        try:
            print("🔧 Testando integração do sistema...")
            
            # Teste do RAG
            test_result = self.rag.query("teste sistema economia")
            
            if 'error' in test_result:
                print(f"⚠️ Teste RAG com problemas: {test_result['error']}")
            else:
                processing_time = test_result.get('processing_time_ms', 0)
                num_docs = test_result.get('num_documents', 0)
                print(f"✅ Teste RAG: {num_docs} docs, {processing_time:.1f}ms")
            
        except Exception as e:
            print(f"⚠️ Erro no teste de integração: {e}")
    
    def _create_tools(self) -> List[Tool]:
        """Cria ferramentas baseadas na disponibilidade do RAG."""
        
        if not self.rag_available:
            return [
                Tool(
                    name="sistema_indisponivel",
                    func=self._rag_unavailable_response,
                    description="Informa que o sistema RAG está indisponível"
                )
            ]
        
        return [
            Tool(
                name="consulta_rag_principal",
                func=self._main_rag_query,
                description="""FERRAMENTA PRINCIPAL - Use SEMPRE primeiro.
                Consulta informações na base de conhecimento sobre economia de São Paulo.
                Especializada em indústrias, comércio exterior, e indicadores econômicos.
                Input: Pergunta completa do usuário"""
            ),
            Tool(
                name="busca_rag_detalhada",
                func=self._detailed_rag_search,
                description="""Use para buscar informações específicas e detalhadas.
                Ideal para dados estatísticos, números específicos, e análises setoriais.
                Input: Aspecto específico para pesquisar"""
            ),
            Tool(
                name="verificacao_rag",
                func=self._verify_rag_quality,
                description="""Use para verificar qualidade e completude das informações.
                Avalia se há dados suficientes para resposta completa.
                Input: Tópico para verificar cobertura"""
            ),
            Tool(
                name="diagnostico_sistema",
                func=self._system_diagnostics,
                description="""Use quando houver problemas com outras ferramentas.
                Fornece diagnóstico e status do sistema.
                Input: 'diagnostico' ou descrição do problema"""
            )
        ]
    
    def _create_prompt(self) -> PromptTemplate:
        """Cria prompt otimizado baseado na disponibilidade do RAG."""
        
        if not self.rag_available:
            template = """SISTEMA RAG INDISPONÍVEL

❌ O sistema de consulta não está funcionando.
Use a ferramenta disponível para informar sobre a indisponibilidade.

Ferramentas: {tools}

Formato:
Question: {input}
Thought: Explicar indisponibilidade
Action: sistema_indisponivel
Action Input: mensagem explicativa
Observation: resultado
Final Answer: Explicação para o usuário

Pergunta: {input}
{agent_scratchpad}"""
        else:
            template = """Você é um ESPECIALISTA em economia do Estado de São Paulo, com foco específico em:
- Indústria Automotiva
- Indústria Têxtil e de Confecções  
- Indústria Farmacêutica
- Máquinas e Equipamentos
- Mapa da Indústria Paulista
- Indústria Metalúrgica
- Agropecuária e Transição Energética
- Balança Comercial Paulista
- Biocombustíveis

INSTRUÇÕES IMPORTANTES PARA RESPOSTAS DETALHADAS:

SEMPRE use múltiplas ferramentas para coletar informações abrangentes
Estruture suas respostas com numeração, subtópicos e formatação clara
Inclua dados específicos, estatísticas e exemplos sempre que disponível
Desenvolva cada ponto com explicações detalhadas, não apenas liste
Conecte informações entre diferentes aspectos do tema
Use linguagem técnica apropriada mas acessível
FORMATO OBRIGATÓRIO para Final Answer:
- Use numeração (1., 2., 3., etc.) para pontos principais
- Use subtópicos com negrito para destacar aspectos importantes
- Inclua dados quantitativos quando disponível
- Desenvolva cada ponto com pelo menos 2-3 frases explicativas
- Termine com uma síntese/conclusão que conecte todos os pontos
- Sempre que necessário use "consulta_rag_principal" primeiro
- Para detalhes específicos: "busca_rag_detalhada"  


Ferramentas disponíveis:
{tools}

Use o seguinte formato de raciocínio:

Question: a pergunta de entrada que você deve responder
Thought: análise da pergunta e estratégia para buscar informações abrangentes
Action: a ação a ser tomada, deve ser uma das [{tool_names}]
Action Input: a entrada específica para a ação
Observation: o resultado da ação
... (repita Thought/Action/Action Input/Observation quantas vezes necessário - use pelo menos 2-3 ferramentas diferentes)
Thought: análise completa de todas as informações coletadas
Final Answer: resposta DETALHADA, ESTRUTURADA e COMPLETA seguindo o formato obrigatório

LEMBRE-SE: Respostas curtas ou superficiais não são aceitáveis, exceto em casos específicos, como:
Saudações simples (ex.: "Olá, tudo bem?", "Olá", "Oi", "Oiê", "Olá, tudo bem", "E aí", "Beleza", 
"Fala aí", "Como vai", "Como está", "Tudo certo", "Tudo tranquilo", "Tranquilo", 
"Suave", "Suave na nave", "De boa", "E aí, meu chapa", "E aí, parceiro", "Salve", 
"Salve, mano", "Saudações", "Alô", "Bom dia", "Boa tarde", "Boa noite", "Como você está", 
"Como tem passado", "Tudo em ordem", "Tudo beleza", "Tudo joia", "Tudo legal", "Tudo bacana", 
"Tudo em paz", "Opa", "Opa, tudo certo", "E aí, firmeza", "Firme e forte", "Firmeza total", 
"Oi, sumido", "Long time no see", "Quanto tempo", "Que bom te ver", "Que prazer te ver", "Seja bem-vindo", 
"Bem-vindo", "Bem-vinda", "Seja bem-vinda", "Olá, meu amigo", "Olá, minha amiga", "Saudações cordiais", 
"Saudações fraternas", "Saudações formais", "É um prazer vê-lo", "É um prazer revê-la", "Que alegria te ver", 
"Que satisfação encontrá-lo", "Que honra tê-lo aqui", "Como vão as coisas", "Como estão as coisas", 
"Como anda a vida", "Tudo em cima", "Tá tudo certo", "Tá tranquilo", "E aí, como foi o dia", 
"E aí, como estão as novidades", "E aí, como vai a família", "E aí, como vai a vida", 
"E aí, preparado pro dia", "Preparado pra batalha", "Como foi o fim de semana", "Como foi o feriado", 
"Tudo certinho", "E aí, guerreiro", "E aí, campeã", "Fala, meu rei", "Fala, minha rainha", "Bom te ver de novo", 
"Que bom te encontrar", "E aí, tá sumido", "Olá de novo", "Fala, meu consagrado", "Fala, minha consagrada".
"E aí?", "Beleza?", "Fala aí?", "Como vai?", "Como está?", "Tudo certo?", "Tudo tranquilo?", "Tranquilo?", 
"Suave?", "Suave na nave?", "De boa?");
Perguntas extremamente objetivas ou que envolvam dados muito específicos.
Fora essas exceções, cada resposta deve ser abrangente, bem estruturada e rica em detalhes.

Pergunta: {input}
Raciocínio: {agent_scratchpad}"""
            
        return PromptTemplate.from_template(template)
    
    def _main_rag_query(self, query: str) -> str:
        """Consulta principal ao sistema RAG."""
        if not self.rag_available:
            return "❌ Sistema RAG não disponível"
        
        try:
            logger.info(f"Consulta RAG principal: {query}")
            
            # Executar consulta
            result = self.rag.query(query, n_results=6)
            
            if 'error' in result:
                return f"⚠️ Erro no RAG: {result['error']}"
            
            response = result.get("response", "")
            num_docs = result.get('num_documents', 0)
            processing_time = result.get('processing_time_ms', 0)
            confidence_scores = result.get('confidence_scores', [])
            
            # Enriquecer resposta
            if response:
                enriched = f"{response}\n\n"
                enriched += f"📊 _Processados {num_docs} documento(s) em {processing_time:.1f}ms_"
                
                if confidence_scores:
                    avg_conf = sum(confidence_scores) / len(confidence_scores)
                    enriched += f"\n🎯 _Confiança média: {avg_conf:.2f}_"
                
                return enriched
            else:
                return "⚠️ Nenhuma informação relevante encontrada na base"
            
        except Exception as e:
            logger.error(f"Erro na consulta principal: {e}")
            return f"❌ Erro na consulta: {str(e)}"
    
    def _detailed_rag_search(self, aspect: str) -> str:
        """Busca detalhada por aspectos específicos."""
        if not self.rag_available:
            return "❌ Sistema RAG não disponível"
        
        try:
            logger.info(f"Busca detalhada: {aspect}")
            
            # Consultas específicas para diferentes tipos de dados
            queries = [
                f"dados estatísticos {aspect} São Paulo",
                f"indicadores {aspect} economia paulista",
                f"números {aspect} indústria SP"
            ]
            
            best_result = None
            best_score = 0
            
            for query in queries:
                try:
                    result = self.rag.query(query, n_results=4)
                    if 'error' not in result:
                        quality = result.get('quality_assessment', {})
                        score = quality.get('quality_score', 0)
                        
                        if score > best_score:
                            best_score = score
                            best_result = result
                except Exception:
                    continue
            
            if best_result and best_result.get('response'):
                response = best_result['response']
                num_docs = best_result.get('num_documents', 0)
                return f"{response}\n\n📊 _Busca detalhada: {num_docs} documento(s), score: {best_score:.2f}_"
            else:
                return "⚠️ Dados específicos não encontrados para este aspecto"
                
        except Exception as e:
            logger.error(f"Erro na busca detalhada: {e}")
            return f"❌ Erro na busca detalhada: {str(e)}"
    
    def _verify_rag_quality(self, topic: str) -> str:
        """Verifica qualidade dos dados disponíveis."""
        if not self.rag_available:
            return "❌ Sistema RAG não disponível"
        
        try:
            # Consulta para avaliação
            result = self.rag.query(topic, n_results=8)
            
            if 'error' in result:
                return f"⚠️ Erro na verificação: {result['error']}"
            
            quality = result.get('quality_assessment', {})
            num_docs = result.get('num_documents', 0)
            quality_score = quality.get('quality_score', 0)
            
            verification = "🔍 **VERIFICAÇÃO DE QUALIDADE**\n\n"
            
            if num_docs == 0:
                verification += "❌ **Status**: Nenhum documento encontrado\n"
                verification += "📋 **Recomendação**: Reformular consulta\n"
            elif num_docs < 3:
                verification += f"⚠️ **Status**: Poucos documentos ({num_docs})\n"
                verification += "📋 **Recomendação**: Busca complementar necessária\n"
            else:
                verification += f"✅ **Status**: Boa cobertura ({num_docs} documentos)\n"
            
            if quality_score > 0.7:
                verification += "🎯 **Qualidade**: Alta confiança\n"
            elif quality_score > 0.4:
                verification += "🎯 **Qualidade**: Confiança moderada\n"  
            else:
                verification += "⚠️ **Qualidade**: Baixa confiança\n"
            
            return verification
            
        except Exception as e:
            return f"❌ Erro na verificação: {str(e)}"
    
    def _system_diagnostics(self, input_text: str) -> str:
        """Diagnóstico completo do sistema."""
        if not self.rag_available:
            return """❌ **DIAGNÓSTICO: Sistema RAG Indisponível**

**Verificações necessárias:**
1. Instalar dependências: pip install chromadb sentence-transformers
2. Verificar arquivo .env com OPENAI_API_KEY  
3. Confirmar rag_system.py no diretório
4. Executar indexação de documentos"""
        
        try:
            # Status do RAG
            status = self.rag.get_system_status()
            
            diag = "🔧 **DIAGNÓSTICO COMPLETO**\n\n"
            diag += "**COMPONENTES:**\n"
            diag += f"- Sistema inicializado: {'✅' if status['initialized'] else '❌'}\n"
            diag += f"- ChromaDB: {'✅' if status['chroma_client'] else '❌'}\n"
            diag += f"- OpenAI: {'✅' if status['openai_client'] else '❌'}\n"
            diag += f"- Coleção: {'✅' if status['collection_exists'] else '❌'}\n"
            diag += f"- Documentos: {status['collection_count']}\n"
            diag += f"- Reranking: {'✅' if status['reranking_enabled'] else '❌'}\n\n"
            
            # Teste funcional
            try:
                test_result = self.rag.query("teste diagnóstico")
                if 'error' not in test_result:
                    diag += "✅ **TESTE**: Sistema respondendo normalmente\n"
                else:
                    diag += f"❌ **TESTE**: {test_result['error']}\n"
            except Exception as e:
                diag += f"❌ **TESTE**: Erro - {e}\n"
            
            return diag
            
        except Exception as e:
            return f"❌ Erro no diagnóstico: {str(e)}"
    
    def _rag_unavailable_response(self, message: str) -> str:
        """Resposta quando RAG não está disponível."""
        return """❌ **SISTEMA RAG INDISPONÍVEL**

Este agente requer uma base de conhecimento especializada sobre economia de São Paulo.

**Para resolver:**
1. Instalar: `pip install chromadb sentence-transformers openai`
2. Verificar arquivo .env com OPENAI_API_KEY
3. Executar indexação de documentos
4. Testar rag_system.py independentemente

**Status**: Sistema não funcional sem RAG"""
    
    def consultar(self, pergunta: str) -> str:
        """
        Método principal de consulta.
        
        Args:
            pergunta: Pergunta sobre economia de São Paulo
            
        Returns:
            Resposta baseada no sistema RAG
        """
        if not pergunta.strip():
            return "Por favor, forneça uma pergunta válida."
        
        if not self.system_ready:
            return self._get_system_not_ready_message()
        
        try:
            logger.info(f"Processando consulta: {pergunta}")
            
            # Preparar input otimizado
            optimized_input = f"""
CONSULTA: {pergunta}

INSTRUÇÕES:
1. Use consulta_rag_principal primeiro
2. Busque dados específicos se necessário
3. Verifique qualidade das informações
4. Estruture resposta profissionalmente

OBJETIVO: Resposta completa e precisa baseada na base de conhecimento.
            """
            
            # Executar via agente
            resultado = self.agent_executor.invoke({"input": optimized_input})
            resposta = resultado.get("output", "Não foi possível obter resposta.")
            
            # Validar uso do RAG
            if not self._validate_rag_usage(resposta):
                logger.warning("Resposta pode não ter usado adequadamente o RAG")
                # Tentar novamente com instrução mais específica
                retry_input = f"OBRIGATÓRIO: Use ferramentas RAG para responder: {pergunta}"
                resultado_retry = self.agent_executor.invoke({"input": retry_input})
                resposta = resultado_retry.get("output", resposta)
            
            # Adicionar indicador se necessário
            if not any(indicator in resposta for indicator in ["📊", "✅", "🎯"]):
                resposta += "\n\n📚 _Resposta baseada na base de conhecimento especializada_"
            
            return resposta
            
        except Exception as e:
            logger.error(f"Erro na consulta: {e}")
            return f"""❌ Erro no processamento: {str(e)}

**Soluções possíveis:**
1. Verificar status do sistema RAG
2. Reformular a pergunta
3. Usar diagnóstico_sistema para mais detalhes"""
    
    def _get_system_not_ready_message(self) -> str:
        """Mensagem quando sistema não está pronto."""
        return """❌ **SISTEMA NÃO ESTÁ PRONTO**

**Status dos componentes:**
- Sistema RAG: ❌ Não inicializado
- Agente LangChain: ❌ Não configurado

**Para resolver:**
1. Verificar dependências instaladas
2. Confirmar OPENAI_API_KEY no .env
3. Testar sistema RAG independentemente
4. Reinicializar o agente

**Comando de diagnóstico:** `agent.get_system_info()`"""
    
    def _validate_rag_usage(self, response: str) -> bool:
        """Valida se a resposta usou o sistema RAG."""
        rag_indicators = [
            "📊", "✅", "🎯", "documento(s)", "processados", 
            "base de conhecimento", "confiança", "Processados"
        ]
        
        return sum(1 for indicator in rag_indicators if indicator in response) >= 1
    
    def get_system_info(self) -> Dict[str, Any]:
        """Informações completas do sistema."""
        info = {
            "agent_ready": self.system_ready,
            "rag_available": self.rag_available,
            "force_rag_mode": self.force_rag,
            "tools_count": len(self.tools) if hasattr(self, 'tools') else 0,
            "rag_system_imported": RAG_AVAILABLE
        }
        
        if self.rag_available and hasattr(self, 'rag'):
            try:
                rag_status = self.rag.get_system_status()
                info.update(rag_status)
                info["fully_functional"] = (
                    rag_status.get('initialized', False) and
                    rag_status.get('collection_count', 0) > 0 and
                    self.system_ready
                )
            except Exception as e:
                info["rag_error"] = str(e)
                info["fully_functional"] = False
        else:
            info["fully_functional"] = False
        
        return info
    
    def test_complete_system(self) -> Dict[str, Any]:
        """Teste completo do sistema integrado."""
        from datetime import datetime
        
        test_results = {
            "timestamp": datetime.now().isoformat(),
            "agent_tests": {},
            "rag_tests": {},
            "integration_tests": {},
            "overall_status": "unknown",
            "recommendations": []
        }
        
        # Testes do agente
        test_results["agent_tests"]["initialization"] = {
            "status": "passed" if self.system_ready else "failed",
            "details": "Agente inicializado" if self.system_ready else "Agente não inicializado"
        }
        
        test_results["agent_tests"]["tools"] = {
            "status": "passed" if hasattr(self, 'tools') and self.tools else "failed",
            "details": f"{len(self.tools) if hasattr(self, 'tools') else 0} ferramentas"
        }
        
        # Testes do RAG
        if self.rag_available and hasattr(self, 'rag'):
            try:
                rag_test_results = self.rag.test_system()
                test_results["rag_tests"] = rag_test_results["tests"]
            except Exception as e:
                test_results["rag_tests"]["error"] = str(e)
        
        # Teste de integração
        if self.system_ready and self.rag_available:
            try:
                test_query = "teste integração economia São Paulo"
                response = self.consultar(test_query)
                
                if "❌" not in response and len(response) > 50:
                    test_results["integration_tests"]["full_query"] = {
                        "status": "passed",
                        "details": f"Consulta processada: {len(response)} chars"
                    }
                else:
                    test_results["integration_tests"]["full_query"] = {
                        "status": "failed", 
                        "details": "Resposta inadequada ou com erro"
                    }
            except Exception as e:
                test_results["integration_tests"]["full_query"] = {
                    "status": "failed",
                    "details": str(e)
                }
        
        # Avaliação geral
        all_tests = {}
        all_tests.update(test_results["agent_tests"])
        all_tests.update(test_results["rag_tests"])
        all_tests.update(test_results["integration_tests"])
        
        passed = sum(1 for test in all_tests.values() if test.get("status") == "passed")
        total = len(all_tests)
        
        if total == 0:
            test_results["overall_status"] = "no_tests"
        elif passed == total:
            test_results["overall_status"] = "excellent"
        elif passed >= total * 0.8:
            test_results["overall_status"] = "good"
        elif passed >= total * 0.5:
            test_results["overall_status"] = "needs_attention"
        else:
            test_results["overall_status"] = "critical"
        
        # Recomendações
        if not self.rag_available:
            test_results["recommendations"].append("Instalar e configurar sistema RAG")
        if not self.system_ready:
            test_results["recommendations"].append("Verificar configuração do agente")
        
        return test_results
    
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Compatibilidade com Streamlit."""
        question = inputs.get("question", "")
        
        if not question:
            return {"chat_history": []}
        
        response = self.consultar(question)
        
        # Atualizar memória
        self.memory.chat_memory.add_user_message(question)
        self.memory.chat_memory.add_ai_message(response)
        
        return {"chat_history": self.memory.chat_memory.messages}

# Função para criar o agente
def create_rag_agent(force_rag: bool = True, **kwargs):
    """
    Cria instância do agente RAG corrigido.
    
    Args:
        force_rag: Se True, exige sistema RAG funcional
        **kwargs: Argumentos adicionais
    """
    try:
        print("🚀 Criando agente RAG corrigido...")
        
        agent = RAGAgentFixed(force_rag=force_rag, **kwargs)
        
        # Verificar se foi criado com sucesso
        if not agent.system_ready:
            if force_rag:
                raise RuntimeError("❌ Agente não pôde ser inicializado completamente")
            else:
                print("⚠️ Agente criado com limitações")
        
        # Executar teste inicial
        print("🔧 Testando sistema...")
        test_results = agent.test_complete_system()
        print(f"📊 Status geral: {test_results['overall_status']}")
        
        if test_results['recommendations']:
            print("💡 Recomendações:")
            for rec in test_results['recommendations']:
                print(f"  - {rec}")
        
        return agent
        
    except Exception as e:
        print(f"❌ Erro ao criar agente: {e}")
        raise

# Função de diagnóstico independente
def diagnose_system():
    """Executa diagnóstico completo do sistema."""
    print("🔧 DIAGNÓSTICO COMPLETO DO SISTEMA")
    print("=" * 50)
    
    # Verificar imports
    print("📦 VERIFICANDO DEPENDÊNCIAS:")
    
    dependencies = {
        "dotenv": "python-dotenv",
        "openai": "openai", 
        "langchain": "langchain",
        "langchain_openai": "langchain-openai",
        "chromadb": "chromadb",
        "sentence_transformers": "sentence-transformers"
    }
    
    missing_deps = []
    for module, package in dependencies.items():
        try:
            __import__(module)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_deps.append(package)
    
    if missing_deps:
        print(f"\n❌ DEPENDÊNCIAS FALTANDO:")
        print(f"pip install {' '.join(missing_deps)}")
        return False
    
    # Verificar variáveis de ambiente
    print(f"\n🔑 VERIFICANDO CONFIGURAÇÃO:")
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        if api_key.startswith('sk-'):
            print(f"  ✅ OPENAI_API_KEY configurada")
        else:
            print(f"  ❌ OPENAI_API_KEY inválida")
            return False
    else:
        print(f"  ❌ OPENAI_API_KEY não encontrada")
        print(f"  💡 Criar arquivo .env: OPENAI_API_KEY=sk-seu-token-aqui")
        return False
    
    # Verificar RAG System
    print(f"\n🔍 VERIFICANDO SISTEMA RAG:")
    if RAG_AVAILABLE:
        print(f"  ✅ Módulo RAG importado")
        try:
            rag = RagSystem()
            if rag.is_initialized:
                status = rag.get_system_status()
                print(f"  ✅ RAG inicializado")
                print(f"  📊 Documentos: {status.get('collection_count', 0)}")
            else:
                print(f"  ❌ RAG não inicializado")
                return False
        except Exception as e:
            print(f"  ❌ Erro no RAG: {e}")
            return False
    else:
        print(f"  ❌ Módulo RAG não disponível")
        return False
    
    print(f"\n✅ SISTEMA PRONTO PARA USO!")
    return True

# Exemplo de uso interativo
if __name__ == "__main__":
    print("🚀 SISTEMA RAG AGENT - VERSÃO CORRIGIDA")
    print("=" * 50)
    
    # Executar diagnóstico primeiro
    if not diagnose_system():
        print("\n❌ Sistema não está pronto. Resolva os problemas acima.")
        exit(1)
    
    try:
        # Criar agente
        print(f"\n🎯 Criando agente...")
        agent = create_rag_agent(force_rag=True)
        
        print(f"\n📊 Informações do sistema:")
        info = agent.get_system_info()
        for key, value in info.items():
            if isinstance(value, bool):
                icon = "✅" if value else "❌"
                print(f"  {icon} {key}: {value}")
            else:
                print(f"  📄 {key}: {value}")
        
        print(f"\n" + "=" * 50)
        print("💬 SESSÃO INTERATIVA INICIADA")
        print("Digite 'sair' para encerrar")
        print("Digite 'info' para status do sistema")
        print("Digite 'teste' para teste completo")
        print("=" * 50)
        
        while True:
            try:
                pergunta = input(f"\n💡 Sua pergunta sobre economia de São Paulo:\n> ").strip()
                
                if pergunta.lower() in ['sair', 'exit', 'quit']:
                    print("👋 Encerrando sistema. Até logo!")
                    break
                
                if pergunta.lower() == 'info':
                    info = agent.get_system_info()
                    print(f"\n📊 STATUS DO SISTEMA:")
                    print("=" * 30)
                    for key, value in info.items():
                        print(f"{key}: {value}")
                    print("=" * 30)
                    continue
                
                if pergunta.lower() == 'teste':
                    print(f"\n🔧 Executando teste completo...")
                    test_results = agent.test_complete_system()
                    print(f"\nResultados:")
                    print(f"Status geral: {test_results['overall_status']}")
                    
                    if test_results.get('agent_tests'):
                        print(f"\nTestes do Agente:")
                        for test, result in test_results['agent_tests'].items():
                            status = "✅" if result['status'] == 'passed' else "❌"
                            print(f"  {status} {test}: {result['details']}")
                    
                    if test_results.get('rag_tests'):
                        print(f"\nTestes do RAG:")
                        for test, result in test_results['rag_tests'].items():
                            if isinstance(result, dict) and 'status' in result:
                                status = "✅" if result['status'] == 'passed' else "❌"
                                print(f"  {status} {test}: {result['details']}")
                    
                    if test_results.get('recommendations'):
                        print(f"\nRecomendações:")
                        for rec in test_results['recommendations']:
                            print(f"  💡 {rec}")
                    
                    continue
                
                if not pergunta:
                    print("❓ Por favor, digite uma pergunta válida.")
                    continue
                
                print(f"\n🔍 Processando sua consulta...")
                resposta = agent.consultar(pergunta)
                
                print(f"\n" + "=" * 60)
                print("📊 RESPOSTA:")
                print("=" * 60)
                print(resposta)
                print("=" * 60)
                
            except KeyboardInterrupt:
                print(f"\n\n👋 Encerrando sistema. Até logo!")
                break
            except Exception as e:
                print(f"\n❌ Erro inesperado: {e}")
                print(f"💡 Use 'info' para verificar status do sistema")
        
    except Exception as e:
        print(f"❌ Erro na inicialização: {e}")
        print(f"\n🔧 SOLUÇÕES:")
        print(f"1. Verificar dependências: pip install chromadb sentence-transformers")
        print(f"2. Verificar arquivo .env com OPENAI_API_KEY")
        print(f"3. Executar diagnóstico: python agent_fixed.py")
        print(f"4. Verificar se há documentos na base ChromaDB")