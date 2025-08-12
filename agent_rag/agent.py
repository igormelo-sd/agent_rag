# agent.py 
import os
import logging
from typing import Dict, Any, List, Tuple

# Carregar variáveis do arquivo .env
from dotenv import load_dotenv
load_dotenv()

# Desabilitar LangSmith (opcional - remove warnings)
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

# Imports corretos para a nova API
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain.schema import HumanMessage, AIMessage

# LangGraph imports para memória
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from typing_extensions import TypedDict

# Import correto considerando que estamos na pasta rag
try:
    from rag_system import RagSystem
    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False
    print(f"⚠️ Aviso: RagSystem não disponível: {e}")

# Verificar se ChromaDB está disponível
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️ Aviso: ChromaDB não disponível")

# Configurar logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langsmith")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Estado para o LangGraph
class ConversationState(TypedDict):
    messages: List[Dict[str, str]]
    last_user_message: str
    last_ai_message: str


class RAGAgentReact:
    """
    Agente RAG aprimorado com tratamento robusto de erros e fallback.
    CORREÇÃO: Simplificação do prompt e controle de iterações para evitar loops.
    Atualizado com LangGraph Memory System.
    """
    
    def __init__(self, openai_api_key: str = None):
        """
        Inicializa o agente RAG com configurações aprimoradas e tratamento de erro.
        """
        # Carregar do .env se não fornecida
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        else:
            # Verificar se foi carregada do .env
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY não encontrada. Verifique se:\n"
                    "1. O arquivo .env existe na raiz do projeto\n"
                    "2. Contém: OPENAI_API_KEY=sk-seu-token-aqui\n"
                    "3. O python-dotenv está instalado: pip install python-dotenv"
                )
            print(f"✅ API Key carregada do .env: {api_key[:10]}...")
        
        # Inicialização segura do sistema RAG
        self.rag_available = False
        self.rag_status = "not_initialized"
        
        if RAG_AVAILABLE and CHROMADB_AVAILABLE:
            try:
                print("🔄 Inicializando sistema RAG...")
                self.rag = RagSystem()
                
                # Testar a conexão do sistema RAG
                system_info = self.rag.get_system_info()
                
                if system_info.get('rag_available', False):
                    self.rag_available = True
                    self.rag_status = "active"
                    print(f"✅ Sistema RAG inicializado: {system_info.get('rag_status', 'Status desconhecido')}")
                else:
                    self.rag_status = f"initialization_failed: {system_info.get('rag_status', 'Falha desconhecida')}"
                    print(f"⚠️ Sistema RAG com problemas: {system_info.get('rag_status', 'Falha desconhecida')}")
                    
            except Exception as e:
                logger.error(f"Erro ao inicializar RAG: {e}")
                self.rag_status = f"error: {str(e)}"
                print(f"❌ Erro na inicialização do RAG: {e}")
        elif not CHROMADB_AVAILABLE:
            self.rag_status = "chromadb_not_available"
            print("❌ ChromaDB não disponível - instale com: pip install chromadb")
        else:
            self.rag_status = "rag_system_not_available"
            print("❌ RagSystem não disponível")
        
        # Configuração do LLM com parâmetros otimizados
        self.llm = ChatOpenAI(
            temperature=0.3,  # Reduzido para mais consistência
            model="gpt-4o",
            max_tokens=8000,   # Reduzido para evitar timeouts
            top_p=0.9,
        )
        
        # MUDANÇA PRINCIPAL: Substituir ConversationBufferMemory por LangGraph Memory
        self.memory_saver = MemorySaver()
        self.thread_id = "main_conversation"  # ID único para a thread de conversação
        
        # Inicializar estado da conversação
        self.conversation_state: ConversationState = {
            "messages": [],
            "last_user_message": "",
            "last_ai_message": ""
        }
        
        # Definir ferramentas simplificadas
        self.tools = self._create_simplified_tools()
        
        # Criar prompt simplificado
        self.prompt = self._create_simplified_prompt()
        
        # Criar agente usando create_react_agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        # CORREÇÃO PRINCIPAL: Configurações mais restritivas para evitar loops
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,        # REDUZIDO de 5 para 3
            max_execution_time=60,   # REDUZIDO de 120 para 60 segundos
            return_intermediate_steps=False,  # Desabilitado para simplicidade
            early_stopping_method="generate"  # Para quando conseguir uma resposta
        )
        
        logger.info(f"Agente RAG inicializado - Status RAG: {self.rag_status}")
    
    def _add_to_memory(self, user_message: str, ai_message: str):
        """Adiciona mensagens à memória usando LangGraph."""
        try:
            # Atualizar estado da conversação
            self.conversation_state["messages"].append({
                "role": "user",
                "content": user_message,
                "timestamp": str(int(os.times().elapsed))
            })
            self.conversation_state["messages"].append({
                "role": "assistant", 
                "content": ai_message,
                "timestamp": str(int(os.times().elapsed))
            })
            
            self.conversation_state["last_user_message"] = user_message
            self.conversation_state["last_ai_message"] = ai_message
            
            # Salvar no MemorySaver
            self.memory_saver.put(
                config={"configurable": {"thread_id": self.thread_id}},
                checkpoint={
                    "state": self.conversation_state,
                    "metadata": {"step": len(self.conversation_state["messages"]) // 2}
                }
            )
            
            logger.info(f"Mensagens adicionadas à memória. Total: {len(self.conversation_state['messages'])}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar na memória: {e}")
    
    def _get_chat_history(self) -> List[Dict[str, str]]:
        """Recupera o histórico de chat da memória."""
        try:
            # Tentar recuperar do MemorySaver
            checkpoint = self.memory_saver.get(
                config={"configurable": {"thread_id": self.thread_id}}
            )
            
            if checkpoint and "state" in checkpoint:
                return checkpoint["state"].get("messages", [])
            else:
                return self.conversation_state["messages"]
                
        except Exception as e:
            logger.error(f"Erro ao recuperar histórico: {e}")
            return self.conversation_state["messages"]
    
    def _format_chat_history_for_prompt(self) -> str:
        """Formata o histórico para incluir no prompt."""
        history = self._get_chat_history()
        if not history:
            return ""
        
        # Pegar apenas as últimas 6 mensagens para evitar prompts muito longos
        recent_history = history[-6:] if len(history) > 6 else history
        
        formatted = []
        for msg in recent_history:
            role = "Human" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content'][:200]}...")  # Truncar para evitar prompts longos
        
        return "\n".join(formatted)
    
    def _create_simplified_tools(self) -> List[Tool]:
        """Cria ferramentas simplificadas para evitar loops."""
        tools = []
        
        if self.rag_available:
            # CORREÇÃO: Apenas uma ferramenta principal para evitar confusão do agente
            tools.append(
                Tool(
                    name="consultar_base_conhecimento",
                    func=self._consultar_rag_direto,
                    description="""FERRAMENTA PRINCIPAL: Consulta a base de conhecimento sobre economia de São Paulo.
                    Use esta ferramenta para responder perguntas sobre:
                    - Indústria (automotiva, têxtil, farmacêutica, metalúrgica, etc.)
                    - Economia do Estado de São Paulo
                    - Dados estatísticos e indicadores
                    - Mapa da Indústria Paulista
                    - Balança Comercial
                    - Agropecuária e outros setores
                    
                    Input: A pergunta exata do usuário
                    Output: Resposta completa baseada na base de conhecimento"""
                )
            )
        else:
            tools.append(
                Tool(
                    name="resposta_geral",
                    func=self._resposta_conhecimento_geral,
                    description="""Use esta ferramenta quando o sistema RAG não estiver disponível.
                    Fornece informações gerais sobre economia de São Paulo.
                    
                    Input: Pergunta do usuário
                    Output: Resposta baseada em conhecimento geral"""
                )
            )
        
        return tools
    
    def _create_simplified_prompt(self) -> PromptTemplate:
        """Cria um prompt simplificado que evita loops infinitos."""
        
        # CORREÇÃO: Definir template base primeiro, depois personalizar
        base_template = """Você é um ESPECIALISTA em economia do Estado de São Paulo.

IMPORTANTE: Para saudações simples (olá, oi, bom dia, etc.) responda diretamente SEM usar ferramentas.

Para outras perguntas sobre economia paulista, use as ferramentas disponíveis.

HISTÓRICO DA CONVERSA:
{chat_history}

Ferramentas disponíveis:
{tools}

Use o seguinte formato:

Question: {input}
Thought: análise da pergunta
Action: escolha uma ferramenta de [{tool_names}]
Action Input: entrada para a ferramenta
Observation: resultado da ferramenta
Thought: análise final
Final Answer: resposta completa e estruturada

{agent_scratchpad}"""
        
        if self.rag_available:
            # Template específico para quando RAG está disponível
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

INSTRUÇÕES PARA RESPOSTAS DETALHADAS:

1. Use a ferramenta disponível para coletar informações abrangentes
2. Estruture suas respostas com numeração, subtópicos e formatação clara
3. Inclua dados específicos, estatísticas e exemplos sempre que disponível
4. Desenvolva cada ponto com explicações detalhadas
5. Use linguagem técnica apropriada mas acessível

FORMATO OBRIGATÓRIO para Final Answer:
- Use numeração (1., 2., 3., etc.) para pontos principais
- Use subtópicos com **negrito** para destacar aspectos importantes
- Inclua dados quantitativos quando disponível
- Desenvolva cada ponto com pelo menos 2-3 frases explicativas

EXCEÇÕES para respostas diretas (SEM usar ferramentas):
- **Saudações**: "Olá", "Oi", "Bom dia", "Boa tarde", "Boa noite", "Tudo bem?", etc.
- **Confirmações**: "Ok", "Entendi", "Certo", "Sim", "Não"
- **Perguntas sobre funcionamento**: "Como você funciona?", "O que você pode fazer?"
- **Despedidas**: "Tchau", "Até logo", "Obrigado"

Para essas exceções, responda diretamente de forma amigável.

HISTÓRICO DA CONVERSA:
{chat_history}

Ferramentas disponíveis:
{tools}

Use o seguinte formato:

Question: {input}
Thought: análise da pergunta e estratégia
Action: escolha uma ferramenta de [{tool_names}]
Action Input: entrada específica para a ferramenta
Observation: resultado da ferramenta
Thought: análise final de todas as informações
Final Answer: resposta DETALHADA, ESTRUTURADA e COMPLETA

{agent_scratchpad}"""
        else:
            # Template para quando RAG não está disponível
            template = """Você é um assistente especializado em economia do Estado de São Paulo.

⚠️ AVISO: Sistema de base de conhecimento não disponível. Respostas baseadas em conhecimento geral.

EXCEÇÕES para respostas diretas (SEM usar ferramentas):
- **Saudações**: "Olá", "Oi", "Bom dia", etc.
- **Confirmações**: "Ok", "Entendi", "Certo"
- **Despedidas**: "Tchau", "Até logo"

Para essas exceções, responda diretamente.

HISTÓRICO DA CONVERSA:
{chat_history}

Ferramentas disponíveis:
{tools}

Use o seguinte formato:

Question: {input}
Thought: análise da pergunta
Action: escolha uma ferramenta de [{tool_names}]
Action Input: entrada para a ferramenta
Observation: resultado da ferramenta
Thought: análise final
Final Answer: resposta com base no conhecimento geral disponível

{agent_scratchpad}"""
        
        return PromptTemplate.from_template(template)
    
    def _consultar_rag_direto(self, query: str) -> str:
        """
        CORREÇÃO: Consulta direta e simplificada do RAG.
        """
        try:
            if not self.rag_available:
                return f"❌ Sistema RAG não disponível. Status: {self.rag_status}"
            
            logger.info(f"Consulta RAG: {query}")
            
            # Usar o método correto baseado no RagSystem fornecido
            resultado = self.rag.query_rag_system(query)
            
            if 'error' in resultado:
                logger.error(f"Erro no RAG: {resultado['error']}")
                return f"⚠️ Erro no sistema: {resultado['error']}"
            
            response = resultado.get("response", "")
            
            if not response or len(response.strip()) < 10:
                return "⚠️ Resposta muito curta ou vazia. Verifique se há documentos na base de dados."
            
            # Adicionar metadados mais detalhados
            retrieved_docs = len(resultado.get('retrieved_documents', []))
            reranked_docs = len(resultado.get('reranked_documents', []))
            confidence = resultado.get('confidence_scores', 'N/A')
            
            metadata_info = f"\n\n📊 _Consulta baseada em {retrieved_docs} documento(s) recuperado(s)"
            if reranked_docs > 0:
                metadata_info += f", {reranked_docs} reranqueado(s)"
            if confidence != 'N/A':
                metadata_info += f" (confiança: {confidence})"
            metadata_info += "._"
            
            return response + metadata_info
            
        except AttributeError as e:
            logger.error(f"Método não encontrado no RAG: {e}")
            return f"❌ Erro: Método de consulta não encontrado no sistema RAG: {str(e)}"
        except Exception as e:
            logger.error(f"Erro na consulta RAG: {e}")
            return f"❌ Erro na consulta: {str(e)}"
    
    def _resposta_conhecimento_geral(self, query: str) -> str:
        """Resposta quando RAG não está disponível."""
        return f"""⚠️ **Sistema de base de conhecimento indisponível**

Pergunta: "{query}"

**Resposta baseada em conhecimento geral:**

São Paulo é o principal centro econômico do Brasil, responsável por cerca de 1/3 do PIB nacional. O estado se destaca em diversos setores:

**Principais Setores:**
- **Indústria Automotiva**: Concentrada no ABC paulista e região de Campinas
- **Indústria Farmacêutica**: Forte presença na região metropolitana
- **Têxtil e Confecções**: Setor tradicional do estado
- **Máquinas e Equipamentos**: Distribuído por várias regiões
- **Agropecuária**: Interior do estado, forte em cana-de-açúcar, café, laranja

**⚠️ IMPORTANTE**: Resposta baseada em conhecimento geral. Para informações precisas, consulte:
- FIESP (Federação das Indústrias do Estado de São Paulo)
- Fundação SEADE
- IBGE

Status do sistema RAG: {self.rag_status}"""
    
    def _is_simple_greeting(self, text: str) -> bool:
        """Verifica se é uma saudação simples que não precisa de ferramentas."""
        greetings = [
            "olá", "oi", "oiê", "ola", "bom dia", "boa tarde", "boa noite",
            "como vai", "tudo bem", "e aí", "salve", "alô", "hello", "hi"
        ]
        text_lower = text.lower().strip()
        return any(greeting in text_lower for greeting in greetings) and len(text_lower) < 20
    
    def consultar(self, pergunta: str) -> str:
        """
        CORREÇÃO PRINCIPAL: Consulta simplificada que evita loops.
        """
        if not pergunta.strip():
            return "Por favor, forneça uma pergunta válida."
        
        try:
            logger.info(f"Processando pergunta: {pergunta}")
            
            # CORREÇÃO: Verificar se é saudação simples
            if self._is_simple_greeting(pergunta):
                resposta = """👋 **Olá! Seja bem-vindo!**

Sou um assistente especializado em economia do Estado de São Paulo. Posso ajudá-lo com informações sobre:

🏭 **Setores Industriais:**
- Indústria Automotiva
- Indústria Têxtil e Confecções
- Indústria Farmacêutica
- Máquinas e Equipamentos
- Indústria Metalúrgica

📊 **Dados Econômicos:**
- Balança Comercial Paulista
- Mapa da Indústria Paulista
- Agropecuária e Transição Energética
- Biocombustíveis

💬 **Como posso ajudar?**
Faça sua pergunta sobre qualquer aspecto da economia paulista!"""
                
                # Adicionar à memória
                self._add_to_memory(pergunta, resposta)
                return resposta
            
            # Preparar input com histórico de chat
            chat_history = self._format_chat_history_for_prompt()
            input_with_history = {
                "input": pergunta,
                "chat_history": chat_history
            }
            
            # Executar com timeout mais restritivo
            resultado = self.agent_executor.invoke(
                input_with_history,
                config={"max_execution_time": 45}  # 45 segundos máximo
            )
            
            resposta = resultado.get("output", "Não foi possível obter uma resposta.")
            
            # CORREÇÃO: Verificar se a resposta é válida
            if "Agent stopped due to iteration limit" in resposta:
                # Fallback direto quando há problema de iteração
                if self.rag_available:
                    logger.warning("Fallback: usando consulta RAG direta")
                    resposta = self._consultar_rag_direto(pergunta)
                else:
                    logger.warning("Fallback: usando conhecimento geral")
                    resposta = self._resposta_conhecimento_geral(pergunta)
            
            # Adicionar à memória
            self._add_to_memory(pergunta, resposta)
            
            return resposta
            
        except Exception as e:
            logger.error(f"Erro ao consultar agente: {e}")
            
            # CORREÇÃO: Fallback robusto em caso de erro
            if self.rag_available:
                try:
                    logger.info("Tentando fallback com RAG direto")
                    resposta = self._consultar_rag_direto(pergunta)
                    self._add_to_memory(pergunta, resposta)
                    return resposta
                except:
                    pass
            
            resposta_erro = f"""❌ **Erro no processamento**

Ocorreu um erro ao processar sua pergunta: {str(e)}

**Possíveis soluções:**
1. Tente reformular a pergunta
2. Verifique se é uma pergunta sobre economia de São Paulo
3. Se o problema persistir, reinicie o sistema

Status do RAG: {self.rag_status}"""
            
            self._add_to_memory(pergunta, resposta_erro)
            return resposta_erro
    
    def get_system_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o status do sistema."""
        info = {
            "rag_available": self.rag_available,
            "rag_status": self.rag_status,
            "tools_count": len(self.tools),
            "agent_ready": hasattr(self, 'agent_executor'),
            "max_iterations": 3,  # Atualizado
            "max_execution_time": 60,  # Atualizado
            "memory_system": "LangGraph MemorySaver",
            "messages_count": len(self._get_chat_history()),
            "chromadb_available": CHROMADB_AVAILABLE
        }
        
        if self.rag_available and hasattr(self, 'rag'):
            try:
                rag_status = self.rag.get_system_info()
                info.update({
                    "rag_detailed_status": rag_status,
                    "reranking_enabled": rag_status.get('reranking_enabled', False),
                    "llm_model": rag_status.get('llm_model', 'unknown')
                })
            except Exception as e:
                info["rag_error"] = str(e)
        
        return info
    
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        CORREÇÃO: Método para compatibilidade com Streamlit simplificado.
        Atualizado para usar LangGraph memory.
        """
        question = inputs.get("question", "")
        
        if not question:
            return {"chat_history": []}
        
        # Obter resposta do agente
        response = self.consultar(question)
        
        # Converter mensagens do formato LangGraph para o formato LangChain
        # para compatibilidade com Streamlit
        langgraph_messages = self._get_chat_history()
        langchain_messages = []
        
        for msg in langgraph_messages:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            else:
                langchain_messages.append(AIMessage(content=msg["content"]))
        
        # Retornar no formato esperado pelo Streamlit
        return {
            "chat_history": langchain_messages,
            "output": response  # Adicionar output direto para compatibilidade
        }
    
    def clear_memory(self):
        """Limpa a memória da conversação."""
        try:
            self.conversation_state = {
                "messages": [],
                "last_user_message": "",
                "last_ai_message": ""
            }
            logger.info("Memória limpa com sucesso")
        except Exception as e:
            logger.error(f"Erro ao limpar memória: {e}")
    
    def run_interactive(self):
        """Executa o loop interativo."""
        print("=== Agente RAG Corrigido - Sistema de Consulta ===")
        print("Especialista em economia do Estado de São Paulo")
        print("Agora com LangGraph Memory System")
        
        # Mostrar status do sistema
        system_info = self.get_system_info()
        print(f"\n📊 **Status do Sistema:**")
        print(f"RAG disponível: {'✅ Sim' if system_info['rag_available'] else '❌ Não'}")
        print(f"Status: {system_info['rag_status']}")
        print(f"ChromaDB disponível: {'✅ Sim' if system_info['chromadb_available'] else '❌ Não'}")
        print(f"Máx iterações: {system_info['max_iterations']}")
        print(f"Timeout: {system_info['max_execution_time']}s")
        print(f"Sistema de memória: {system_info['memory_system']}")
        print(f"Mensagens na memória: {system_info['messages_count']}")
        
        # Mostrar detalhes do RAG se disponível
        if system_info.get('rag_detailed_status'):
            rag_details = system_info['rag_detailed_status']
            print(f"Reranking habilitado: {'✅ Sim' if rag_details.get('reranking_enabled') else '❌ Não'}")
            print(f"Modelo LLM: {rag_details.get('llm_model', 'N/A')}")
        
        print(f"\nDigite 'sair' para encerrar, 'limpar' para limpar histórico, 'status' para ver informações\n")
        
        while True:
            try:
                user_input = input("> ").strip()
                
                if user_input.lower() in ["sair", "exit", "quit"]:
                    print("Encerrando. Até logo!")
                    break
                
                if user_input.lower() in ["limpar", "clear"]:
                    self.clear_memory()
                    print("🧹 Histórico limpo!")
                    continue
                
                if user_input.lower() in ["status", "info"]:
                    info = self.get_system_info()
                    print("\n📊 **Status Atual:**")
                    for key, value in info.items():
                        if key != 'rag_detailed_status':
                            print(f"{key}: {value}")
                    print()
                    continue
                
                if not user_input:
                    continue
                
                print(f"\n🔍 Processando...")
                resposta = self.consultar(user_input)
                
                print(f"\n{'='*60}")
                print("📊 RESPOSTA:")
                print(f"{'='*60}")
                print(f"{resposta}")
                print(f"{'='*60}\n")
                
            except KeyboardInterrupt:
                print("\nEncerrando. Até logo!")
                break
            except Exception as e:
                logger.error(f"Erro no loop: {e}")
                print(f"Erro: {e}\n")


def create_rag_agent():
    """
    CORREÇÃO: Função para criar o agente RAG corrigido.
    """
    try:
        os.environ["ANONYMIZED_TELEMETRY"] = "False"
        
        print("Inicializando agente RAG com LangGraph...")
        agent = RAGAgentReact()
        
        system_info = agent.get_system_info()
        if system_info['rag_available']:
            print("✅ Agente RAG completo inicializado!")
        else:
            print(f"⚠️ Agente em modo limitado - Status: {system_info['rag_status']}")
        
        return agent
        
    except Exception as e:
        print(f"❌ Erro ao inicializar: {e}")
        raise


if __name__ == "__main__":
    try:
        os.environ["ANONYMIZED_TELEMETRY"] = "False"
        agent = RAGAgentReact()
        agent.run_interactive()
        
    except ValueError as e:
        print(f"Erro de configuração: {e}")
    except Exception as e:
        print(f"Erro: {e}")