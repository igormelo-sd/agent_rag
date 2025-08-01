# streamlit_app.py - VERSÃO COM GIF CENTRALIZADO E FUNDO FIXO
import streamlit as st
from dotenv import load_dotenv
from htmlTemplates import (
    load_css, ai_template, human_template, 
    show_centralized_waiting, hide_centralized_waiting,
    get_loading_screen_html, initialize_templates
)
from agent import create_rag_agent
import logging
import os
import signal
import threading
from dataclasses import dataclass
from typing import Literal
import base64
from pathlib import Path
import time

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Message:
    """Class for keeping track of a chat message."""
    origin: Literal["human", "ai"]
    message: str

def apply_background_image():
    """
    Aplica a imagem de fundo personalizada se disponível
    """
    current_dir = Path.cwd()
    search_dirs = [
        current_dir,
        current_dir / "images",
        current_dir / "img", 
        current_dir / "assets",
        current_dir / "static",
    ]
    
    background_path = None
    background_names = ["chat_robot_background.png", "background.png", "bg.png"]
    
    # Procurar imagem de fundo
    for directory in search_dirs:
        if not directory.exists():
            continue
        for name in background_names:
            path = directory / name
            if path.exists() and path.is_file():
                background_path = str(path)
                break
        if background_path:
            break
    
    if background_path:
        try:
            with open(background_path, "rb") as img_file:
                background_b64 = base64.b64encode(img_file.read()).decode()
                
            # Aplicar CSS com fundo personalizado
            background_css = f"""
            <style>
            /* Fundo fixo personalizado */
            .stApp {{
                background: url('data:image/png;base64,{background_b64}') center center fixed !important;
                background-size: cover !important;
                background-repeat: no-repeat !important;
                background-attachment: fixed !important;
            }}
            
            /* Overlay semi-transparente para melhor legibilidade */
            .stApp::before {{
                content: '';
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(255, 255, 255, 0.85);
                z-index: -1;
                pointer-events: none;
            }}
            
            /* Ajustar transparência dos containers */
            .chat-messages-container {{
                background: rgba(255, 255, 255, 0.95) !important;
                backdrop-filter: blur(10px) !important;
            }}
            
            .chat-bubble {{
                background: rgba(255, 255, 255, 0.95) !important;
                backdrop-filter: blur(5px) !important;
            }}
            
            .ai-bubble {{
                background: linear-gradient(135deg, rgba(248, 249, 250, 0.95) 0%, rgba(233, 236, 239, 0.95) 100%) !important;
            }}
            
            .human-bubble {{
                background: linear-gradient(135deg, rgba(0, 123, 255, 0.95) 0%, rgba(0, 86, 179, 0.95) 100%) !important;
            }}
            
            .waiting-bubble {{
                background: linear-gradient(135deg, rgba(255, 243, 205, 0.95) 0%, rgba(255, 234, 167, 0.95) 100%) !important;
            }}
            </style>
            """
            
            st.markdown(background_css, unsafe_allow_html=True)
            logger.info("✅ Fundo personalizado aplicado com sucesso!")
            
        except Exception as e:
            logger.error(f"Erro ao carregar fundo personalizado: {e}")
    else:
        logger.info("ℹ️ Fundo personalizado não encontrado, usando padrão")

def get_gif_base64(gif_name):
    """
    Converte GIF para base64 para incorporar no HTML.
    """
    current_dir = Path.cwd()
    search_dirs = [
        current_dir / "img",
        current_dir / "images", 
        current_dir / "assets",
        current_dir,
    ]
    
    for directory in search_dirs:
        gif_path = directory / gif_name
        if gif_path.exists():
            try:
                with open(gif_path, "rb") as gif_file:
                    encoded = base64.b64encode(gif_file.read()).decode()
                    return f"data:image/gif;base64,{encoded}"
            except Exception as e:
                logger.error(f"Erro ao carregar {gif_name}: {e}")
                continue
    
    logger.warning(f"GIF {gif_name} não encontrado!")
    return None

def show_initial_loading():
    """
    Mostra a tela de carregamento inicial com loading_screen.gif
    """
    loading_html = get_loading_screen_html()
    return st.markdown(loading_html, unsafe_allow_html=True)

def initialize_session_state():
    """
    Inicializa o estado da sessão com tela de loading.
    """
    if "history" not in st.session_state:
        st.session_state.history = []
    
    if "conversation" not in st.session_state:
        # Mostrar tela de loading inicial
        loading_placeholder = st.empty()
        
        with loading_placeholder.container():
            show_initial_loading()
        
        try:
            # Simular tempo de carregamento para mostrar o GIF
            time.sleep(2)
            
            # Inicializar o agente
            st.session_state.conversation = create_rag_agent()
            
            # Aguardar um pouco mais para o usuário ver o loading
            time.sleep(1)
            
            # Limpar a tela de loading
            loading_placeholder.empty()
            
            # Mostrar mensagem de sucesso
            st.success("✅ Agente RAG inicializado com sucesso!")
            time.sleep(1)
            
        except Exception as e:
            loading_placeholder.empty()
            st.error(f"❌ Erro ao inicializar o agente: {str(e)}")
            st.stop()

def extract_message_content(message):
    """
    Extrai o conteúdo de uma mensagem, independente do formato.
    """
    if hasattr(message, 'content'):
        return message.content
    elif hasattr(message, 'message'):
        return message.message
    elif isinstance(message, dict):
        return message.get('content', message.get('message', str(message)))
    else:
        return str(message)

def on_click_callback():
    """
    Callback para processar mensagens do usuário.
    """
    try:
        human_prompt = st.session_state.human_prompt
        
        if not human_prompt or not human_prompt.strip():
            return
        
        # Limpar o campo de entrada imediatamente
        st.session_state.human_prompt = ""
        
        if human_prompt.lower().strip() in ['sair', 'exit', 'quit', 'fechar']:
            st.session_state.history.append(
                Message("ai", "Obrigado por usar o chat! Fechando aplicação...")
            )
            st.balloons()
            
            def stop_server():
                time.sleep(3)
                os.kill(os.getpid(), signal.SIGTERM)

            thread = threading.Thread(target=stop_server)
            thread.daemon = True
            thread.start()
            return

        # Adicionar mensagem do usuário
        st.session_state.history.append(
            Message("human", human_prompt)
        )
        
        # Marcar como processando
        st.session_state.processing_response = True
        
        # Marcar que uma atualização é necessária
        st.session_state.needs_update = True
        
    except Exception as e:
        logger.error(f"Erro ao processar pergunta: {e}")
        st.session_state.history.append(
            Message("ai", f"Erro ao processar sua pergunta: {str(e)}")
        )

def process_ai_response(human_prompt):
    """
    Processa a resposta da IA de forma separada
    """
    try:
        # Obter resposta do agente
        response = st.session_state.conversation({"question": human_prompt})
        
        # Extrair a resposta do agente de forma mais robusta
        ai_response = ""
        
        # Tentar diferentes formas de extrair a resposta
        if isinstance(response, dict):
            # Tentar chaves comuns
            ai_response = response.get('output', response.get('answer', response.get('result', '')))
            
            # Se não encontrou, tentar no chat_history
            if not ai_response and 'chat_history' in response:
                chat_history = response['chat_history']
                if chat_history and len(chat_history) > 0:
                    last_message = chat_history[-1]
                    ai_response = extract_message_content(last_message)
        else:
            # Se não é dict, tentar extrair conteúdo diretamente
            ai_response = extract_message_content(response)
        
        # Limpar a resposta se ainda contém metadados
        if ai_response and 'additional_kwargs' in ai_response:
            # Parece que ainda está vindo com metadados, vamos processar
            import re
            # Extrair apenas o conteúdo entre aspas após 'content='
            match = re.search(r"content='([^']*)'", ai_response)
            if match:
                ai_response = match.group(1)
            else:
                ai_response = "Desculpe, não consegui processar a resposta adequadamente."
        
        # Fallback se ainda não conseguiu extrair
        if not ai_response or ai_response == "":
            ai_response = "Desculpe, não consegui gerar uma resposta adequada."
        
        # Adicionar resposta do agente
        st.session_state.history.append(
            Message("ai", ai_response)
        )
        
    except Exception as e:
        logger.error(f"Erro ao obter resposta da IA: {e}")
        st.session_state.history.append(
            Message("ai", f"Erro ao processar sua pergunta: {str(e)}")
        )
    finally:
        # Marcar como não processando
        st.session_state.processing_response = False

def main():
    """
    Função principal do aplicativo Streamlit.
    """
    load_dotenv()

    st.set_page_config(
        page_title="Chat com Agente RAG",
        page_icon=":books:",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # CSS CORRIGIDO - SOLUÇÃO DEFINITIVA PARA QUADRADO BRANCO
    st.markdown("""
    <style>
    /* ========== CORREÇÃO TOTAL PARA QUADRADO BRANCO ========== */
    /* Remover todos os elementos vazios e invisíveis */
    .stApp > div[data-testid="stAppViewContainer"] > div:empty,
    .stApp > div[data-testid="stAppViewContainer"] > div[style*="height: 0"],
    .stApp > div[data-testid="stAppViewContainer"] > div[style*="width: 0"],
    .stApp iframe[height="0"],
    .stApp iframe[width="0"],
    .stApp div:empty:not(.loading-dots div):not(.status-indicator),
    .element-container:empty,
    .stHtml:empty,
    .stComponentHTML:empty {
        display: none !important;
        visibility: hidden !important;
        position: absolute !important;
        left: -9999px !important;
        top: -9999px !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        border: none !important;
        background: transparent !important;
    }

    /* Limpar containers do Streamlit */
    .stApp > div[data-testid="stAppViewContainer"] {
        overflow: hidden !important;
    }

    /* Esconder componentes HTML problemáticos */
    .stComponentHTML > div:empty,
    .stComponentHTML iframe[width="0"],
    .stComponentHTML iframe[height="0"] {
        display: none !important;
    }

    /* Prevenir elementos flutuantes invisíveis */
    .stApp::after,
    .main::after {
        content: "";
        display: table;
        clear: both;
    }

    /* Forçar limpeza de elementos com dimensões zero */
    [style*="height:0px"], [style*="width:0px"],
    [style*="height: 0px"], [style*="width: 0px"],
    [style*="height:0"], [style*="width:0"] {
        display: none !important;
    }

    /* Reset completo para prevenir interferências */
    .stApp * {
        box-sizing: border-box;
    }
    
    .stApp {
        background: #f8f9fa !important;
    }
    /* ========== FIM DA CORREÇÃO ========== */

    /* Reset e configurações básicas */
    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
        margin: 0 !important;
    }
    
    /* Container principal flexível */
    .chat-container {
        display: flex;
        flex-direction: column;
        height: 100vh;
        overflow: hidden;
        background: transparent;
    }
    
    /* Header fixo */
    .chat-header {
        background: linear-gradient(135deg, rgba(0, 123, 255, 0.95) 0%, rgba(0, 86, 179, 0.95) 100%);
        color: white;
        padding: 15px 20px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        flex-shrink: 0;
        z-index: 100;
        backdrop-filter: blur(10px);
    }
    
    /* Área de mensagens que ocupa o espaço disponível */
    .messages-area {
        flex: 1;
        overflow-y: auto;
        padding: 20px 20px 140px 20px;
        background: transparent;
        position: relative;
        display: flex;
        flex-direction: column;
        min-height: 0;
    }
    
    /* Container das mensagens */
    .messages-container {
        flex: 1;
        min-height: 0;
        width: 100%;
    }
    
    /* Área de entrada fixa na parte inferior */
    .input-area {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-top: 2px solid #e9ecef;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        flex-shrink: 0;
        z-index: 1000;
        backdrop-filter: blur(10px);
    }
    
    /* Melhorar aparência dos inputs */
    .stTextArea textarea {
        border-radius: 15px !important;
        border: 2px solid #007bff !important;
        padding: 15px !important;
        font-size: 16px !important;
        resize: none !important;
        min-height: 60px !important;
        max-height: 200px !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        overflow-y: auto !important;
        transition: all 0.2s ease !important;
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(5px) !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #0056b3 !important;
        box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25) !important;
        outline: none !important;
    }
    
    /* Botão de envio */
    .stButton > button {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 15px 30px !important;
        font-weight: bold !important;
        font-size: 16px !important;
        transition: all 0.3s !important;
        width: 100% !important;
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(40, 167, 69, 0.4) !important;
    }
    
    .stButton > button:disabled {
        background: linear-gradient(135deg, #6c757d 0%, #495057 100%) !important;
        transform: none !important;
        box-shadow: 0 2px 8px rgba(108, 117, 125, 0.3) !important;
    }
    
    /* Animações suaves */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0%, 100% { 
            opacity: 0.7; 
            transform: scale(0.95);
        }
        50% { 
            opacity: 1; 
            transform: scale(1);
        }
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    @keyframes loadingDots {
        0%, 100% { opacity: 0.3; transform: scale(0.8); }
        50% { opacity: 1; transform: scale(1); }
    }
    
    /* Loading dots animation */
    .loading-dots {
        display: inline-flex;
        gap: 3px;
        align-items: center;
    }
    
    .loading-dots div {
        width: 8px;
        height: 8px;
        background-color: #007bff;
        border-radius: 50%;
        animation: loadingDots 1.5s infinite;
    }
    
    .loading-dots div:nth-child(2) { animation-delay: 0.5s; }
    .loading-dots div:nth-child(3) { animation-delay: 1s; }
    
    /* Responsividade */
    @media (max-width: 768px) {
        .chat-header {
            padding: 10px 15px;
        }
        
        .messages-area {
            padding: 15px 15px 120px 15px;
        }
        
        .input-area {
            padding: 15px;
        }
        
        .stTextArea textarea {
            font-size: 14px !important;
            min-height: 50px !important;
            max-height: 150px !important;
        }
        
        .stButton > button {
            padding: 12px 20px !important;
            font-size: 14px !important;
        }
    }
    
    /* Scrollbar personalizada */
    .messages-area::-webkit-scrollbar {
        width: 8px;
    }
    
    .messages-area::-webkit-scrollbar-track {
        background: rgba(241, 241, 241, 0.5);
    }
    
    .messages-area::-webkit-scrollbar-thumb {
        background: rgba(193, 193, 193, 0.8);
        border-radius: 4px;
    }
    
    .messages-area::-webkit-scrollbar-thumb:hover {
        background: rgba(168, 168, 168, 0.9);
    }
    
    /* Esconder elementos padrão do Streamlit */
    .stApp > header {
        display: none !important;
    }
    
    .stApp > .main > div:first-child {
        padding-top: 0 !important;
    }
    
    /* Fade in para novas mensagens */
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Carregar CSS do chat
    load_css()
    
    # Aplicar fundo personalizado
    apply_background_image()
    
    # Inicializar estado da sessão com loading screen
    initialize_session_state()

    # Container principal da aplicação
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Header
    st.markdown('''
    <div class="chat-header">
        <h1 style="margin: 0; font-size: 24px;">🤖 Chat com Agente RAG</h1>
        <p style="margin: 5px 0 0 0; opacity: 0.9;">Converse com inteligência artificial avançada</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Área principal de mensagens
    st.markdown('<div class="messages-area">', unsafe_allow_html=True)
    
    # Container das mensagens
    messages_container = st.container()
    
    with messages_container:
        if st.session_state.history:
            # Verificar se há pergunta pendente para processar
            last_message = st.session_state.history[-1]
            
            # Se a última mensagem é do usuário e estamos processando, mostrar GIF centralizado
            if (last_message.origin == 'human' and 
                st.session_state.get('processing_response', False)):
                
                # Mostrar todas as mensagens até agora
                for chat in st.session_state.history:
                    if chat.origin == 'ai':
                        div = ai_template.replace("{{MSG}}", chat.message)
                    else:
                        div = human_template.replace("{{MSG}}", chat.message)
                    st.markdown(div, unsafe_allow_html=True)
                
                # Mostrar GIF centralizado
                show_centralized_waiting()
                
                # Processar resposta da IA
                process_ai_response(last_message.message)
                
                # Esconder GIF centralizado e marcar como não processando
                hide_centralized_waiting()
                st.session_state.processing_response = False
                st.session_state.needs_update = True
            
            else:
                # Mostrar todas as mensagens normalmente
                for chat in st.session_state.history:
                    if chat.origin == 'ai':
                        div = ai_template.replace("{{MSG}}", chat.message)
                    else:
                        div = human_template.replace("{{MSG}}", chat.message)
                    st.markdown(div, unsafe_allow_html=True)
        else:
            st.markdown('''
            <div style="text-align: center; padding: 50px; color: #6c757d; background: rgba(255, 255, 255, 0.8); border-radius: 15px; backdrop-filter: blur(10px);">
                <h3>🌟 Bem-vindo ao Chat RAG!</h3>
                <p>Faça sua primeira pergunta para começar a conversa.</p>
            </div>
            ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Área de entrada sempre visível na parte inferior
    st.markdown('<div class="input-area">', unsafe_allow_html=True)
    
    # Formulário de entrada
    with st.form("chat_form", clear_on_submit=True):
        # Campo de texto que expande automaticamente
        user_input = st.text_area(
            "Digite sua pergunta:",
            placeholder="Digite sua pergunta aqui... (ou 'sair' para fechar)",
            height=60,
            max_chars=2000,
            key="human_prompt",
            label_visibility="collapsed"
        )
        
        # Contador de caracteres
        char_count = len(user_input) if user_input else 0
        st.markdown(f'<div style="text-align: right; font-size: 12px; color: #6c757d; margin-top: 5px;">{char_count}/2000 caracteres</div>', unsafe_allow_html=True)
        
        # Botões de ação
        col_btn1, col_btn2, col_btn3 = st.columns([3, 1, 1])
        
        with col_btn1:
            submit_button = st.form_submit_button(
                "📤 Enviar Mensagem",
                type="primary",
                use_container_width=True,
                on_click=on_click_callback,
                disabled=st.session_state.get('processing_response', False)
            )
        
        with col_btn2:
            if st.form_submit_button("🔄", help="Recarregar", use_container_width=True):
                st.rerun()
        
        with col_btn3:
            if st.form_submit_button("🗑️", help="Limpar", use_container_width=True):
                st.session_state.history = []
                if hasattr(st.session_state, 'conversation') and hasattr(st.session_state.conversation, 'memory'):
                    st.session_state.conversation.memory.clear()
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # JavaScript otimizado para funcionalidades essenciais
    st.markdown("""
    <script>
    (function() {
        'use strict';
        
        // Remover tela de loading após timeout
        setTimeout(function() {
            const loadingScreen = document.getElementById('loading-screen');
            if (loadingScreen) {
                loadingScreen.style.animation = 'fadeOut 0.5s ease-out forwards';
                setTimeout(() => {
                    if (loadingScreen.parentNode) {
                        loadingScreen.parentNode.removeChild(loadingScreen);
                    }
                }, 500);
            }
        }, 3000);
        
        // Auto-scroll para a última mensagem
        function scrollToBottom() {
            const messagesArea = document.querySelector('.messages-area');
            if (messagesArea) {
                messagesArea.scrollTop = messagesArea.scrollHeight;
            }
        }
        
        // Focar textarea automaticamente
        function focusTextArea() {
            const textarea = document.querySelector('textarea');
            if (textarea && !textarea.disabled) {
                textarea.focus();
            }
        }
        
        // Atalhos de teclado para melhor UX
        document.addEventListener('keydown', function(e) {
            if (e.target.tagName === 'TEXTAREA' && e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                const submitBtn = document.querySelector('button[kind="primary"]');
                if (submitBtn && e.target.value.trim() && !submitBtn.disabled) {
                    submitBtn.click();
                }
            }
            
            // Esc para limpar textarea
            if (e.key === 'Escape' && e.target.tagName === 'TEXTAREA') {
                e.target.value = '';
                e.target.focus();
            }
        });
        
        // Executar funções após carregamento
        setTimeout(() => {
            scrollToBottom();
            focusTextArea();
        }, 1000);
        
        // Observer para auto-scroll em novas mensagens
        const observer = new MutationObserver(() => {
            setTimeout(scrollToBottom, 100);
        });
        
        const messagesArea = document.querySelector('.messages-area');
        if (messagesArea) {
            observer.observe(messagesArea, { childList: true, subtree: true });
        }
        
        // Função para animar entrada de novas mensagens
        function animateNewMessages() {
            const messages = document.querySelectorAll('.chat-row:last-child');
            messages.forEach(message => {
                if (!message.classList.contains('animated')) {
                    message.style.animation = 'fadeIn 0.5s ease-out';
                    message.classList.add('animated');
                }
            });
        }
        
        // Chamar animação periodicamente para novas mensagens
        setInterval(animateNewMessages, 500);
        
        // Melhorar experiência de loading
        function enhanceLoadingExperience() {
            // Adicionar efeito de digitação para mensagens de espera
            const waitingBubbles = document.querySelectorAll('.chat-bubble.ai-bubble');
            waitingBubbles.forEach(bubble => {
                if (bubble.textContent.includes('Processando')) {
                    bubble.style.animation = 'pulse 2s infinite';
                }
            });
        }
        
        // Executar melhorias de loading
        setInterval(enhanceLoadingExperience, 1000);
        
        // Limpar elementos problemáticos que causam quadrado branco
        function cleanupEmptyElements() {
            // Remover iframes vazios
            const emptyIframes = document.querySelectorAll('iframe[width="0"], iframe[height="0"]');
            emptyIframes.forEach(iframe => {
                if (iframe.parentNode) {
                    iframe.parentNode.removeChild(iframe);
                }
            });
            
            // Remover divs vazias problemáticas
            const emptyDivs = document.querySelectorAll('div:empty:not(.loading-dots div):not(.status-indicator)');
            emptyDivs.forEach(div => {
                if (div.parentNode && !div.classList.contains('loading-dots') && !div.classList.contains('status-indicator')) {
                    div.style.display = 'none';
                }
            });
            
            // Remover elementos com dimensões zero
            const zeroDimElements = document.querySelectorAll('[style*="height: 0"], [style*="width: 0"]');
            zeroDimElements.forEach(el => {
                el.style.display = 'none';
            });
        }
        
        // Executar limpeza periodicamente
        setInterval(cleanupEmptyElements, 2000);
        
        // Limpeza inicial
        setTimeout(cleanupEmptyElements, 1000);
        
        // Função para controlar o overlay de espera centralizado
        function manageWaitingOverlay() {
            const overlay = document.getElementById('waiting-overlay');
            if (overlay) {
                // Garantir que o overlay está sempre no topo
                overlay.style.zIndex = '9999';
                
                // Auto-remove após 30 segundos como fallback
                setTimeout(() => {
                    if (overlay && overlay.parentNode) {
                        overlay.style.animation = 'fadeOut 0.3s ease-out forwards';
                        setTimeout(() => {
                            if (overlay.parentNode) {
                                overlay.parentNode.removeChild(overlay);
                            }
                        }, 300);
                    }
                }, 30000);
            }
        }
        
        // Monitorar overlay de espera
        setInterval(manageWaitingOverlay, 1000);
        
    })();
    </script>
    """, unsafe_allow_html=True)

    # Verificar se uma atualização da interface é necessária
    if st.session_state.get('needs_update', False):
        st.session_state.needs_update = False  # Resetar a flag
        st.rerun()

if __name__ == '__main__':
    main()
