# htmlTemplates.py
import streamlit as st
import base64
import os
import html
import re
from pathlib import Path

def load_css():
    """
    Carrega o CSS customizado para o chat com GIF centralizado e fundo fixo.
    """
    css = """
    <style>
    /* ========== CORREÇÃO DEFINITIVA PARA QUADRADO BRANCO ========== */
    /* Esconder todos os elementos problemáticos do Streamlit */
    .stApp iframe[height="0"],
    .stApp iframe[width="0"],
    .stApp div:empty:not(.loading-dots div):not(.status-indicator):not(.chat-icon):not(.inline-gif),
    .element-container:empty,
    .stHtml:empty,
    .stComponentHTML:empty,
    .stComponentHTML > div:empty,
    .stComponentHTML iframe[width="0"],
    .stComponentHTML iframe[height="0"],
    [style*="height:0px"], [style*="width:0px"],
    [style*="height: 0px"], [style*="width: 0px"],
    [style*="height:0"], [style*="width:0"] {
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
        pointer-events: none !important;
    }

    /* Limpar containers vazios específicos */
    .stApp > div[data-testid="stAppViewContainer"] > div:empty {
        display: none !important;
    }

    /* Prevenir elementos flutuantes invisíveis */
    .stApp::after,
    .main::after {
        content: "";
        display: table;
        clear: both;
    }
    /* ========== FIM DA CORREÇÃO ========== */
    
    /* ========== FUNDO FIXO PERSONALIZADO ========== */
    .stApp {
        background: url('data:image/png;base64,{{BACKGROUND_IMAGE}}') center center fixed !important;
        background-size: cover !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
    }
    
    /* Overlay semi-transparente para melhor legibilidade */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(255, 255, 255, 0.85);
        z-index: -1;
        pointer-events: none;
    }
    /* ========== FIM DO FUNDO FIXO ========== */
    
    /* Container principal do chat */
    .chat-messages-container {
        max-height: 70vh;
        overflow-y: auto;
        padding: 10px;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Linhas do chat */
    .chat-row {
        display: flex;
        margin: 10px 0;
        width: 100%;
        animation: slideIn 0.4s ease-out;
        align-items: flex-start;
    }
    
    .row-reverse {
        flex-direction: row-reverse;
    }
    
    /* Bolhas de chat */
    .chat-bubble {
        font-family: "Source Sans Pro", sans-serif, "Segoe UI", "Roboto", sans-serif;
        border: 1px solid transparent;
        padding: 12px 16px;
        margin: 0px 10px;
        max-width: 75%;
        word-wrap: break-word;
        line-height: 1.5;
        font-size: 14px;
        position: relative;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(5px);
    }
    
    /* Bolha da IA */
    .ai-bubble {
        background: linear-gradient(135deg, rgba(248, 249, 250, 0.95) 0%, rgba(233, 236, 239, 0.95) 100%);
        border-radius: 18px 18px 18px 4px;
        color: #2c3e50;
        border-left: 4px solid #007bff;
    }
    
    /* Bolha do usuário */
    .human-bubble {
        background: linear-gradient(135deg, rgba(0, 123, 255, 0.95) 0%, rgba(0, 86, 179, 0.95) 100%);
        color: white;
        border-radius: 18px 18px 4px 18px;
        border-right: 4px solid #004085;
    }
    
    /* ========== GIF DE ESPERA CENTRALIZADO ========== */
    .waiting-overlay {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        background: rgba(0, 0, 0, 0.7) !important;
        backdrop-filter: blur(5px) !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        z-index: 9999 !important;
        animation: fadeInOverlay 0.3s ease-out !important;
    }
    
    .waiting-content {
        background: rgba(255, 255, 255, 0.95) !important;
        padding: 40px !important;
        border-radius: 20px !important;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3) !important;
        text-align: center !important;
        max-width: 400px !important;
        animation: scaleIn 0.4s ease-out !important;
    }
    
    .waiting-gif-large {
        width: 120px !important;
        height: 120px !important;
        border-radius: 50% !important;
        border: 4px solid #007bff !important;
        box-shadow: 0 0 30px rgba(0, 123, 255, 0.5) !important;
        margin-bottom: 20px !important;
        animation: pulseGlow 2s infinite !important;
    }
    
    .waiting-text {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        color: #2c3e50 !important;
        margin-bottom: 10px !important;
    }
    
    .waiting-subtitle {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        font-size: 14px !important;
        color: #6c757d !important;
        line-height: 1.5 !important;
    }
    
    @keyframes fadeInOverlay {
        from {
            opacity: 0;
            backdrop-filter: blur(0px);
        }
        to {
            opacity: 1;
            backdrop-filter: blur(5px);
        }
    }
    
    @keyframes scaleIn {
        from {
            transform: scale(0.8);
            opacity: 0;
        }
        to {
            transform: scale(1);
            opacity: 1;
        }
    }
    
    @keyframes pulseGlow {
        0%, 100% {
            transform: scale(1);
            box-shadow: 0 0 30px rgba(0, 123, 255, 0.5);
        }
        50% {
            transform: scale(1.05);
            box-shadow: 0 0 40px rgba(0, 123, 255, 0.8);
        }
    }
    /* ========== FIM DO GIF CENTRALIZADO ========== */
    
    /* Bolha de espera com animação especial (para mensagens inline) */
    .waiting-bubble {
        background: linear-gradient(135deg, rgba(255, 243, 205, 0.95) 0%, rgba(255, 234, 167, 0.95) 100%);
        border-left: 4px solid #ffc107;
        animation: waitingPulse 2s infinite;
    }
    
    @keyframes waitingPulse {
        0%, 100% { 
            opacity: 0.8; 
            transform: scale(0.98);
            box-shadow: 0 2px 8px rgba(255, 193, 7, 0.3);
        }
        50% { 
            opacity: 1; 
            transform: scale(1);
            box-shadow: 0 4px 15px rgba(255, 193, 7, 0.5);
        }
    }
    
    /* Ícones do chat */
    .chat-icon {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        margin: 5px;
        flex-shrink: 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        border: 2px solid #fff;
        object-fit: cover;
    }
    
    /* Ícones GIF com animação especial */
    .gif-icon {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        margin: 5px;
        flex-shrink: 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        border: 3px solid #fff;
        object-fit: cover;
        animation: gifGlow 3s infinite;
    }
    
    @keyframes gifGlow {
        0%, 100% { 
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        50% { 
            box-shadow: 0 4px 20px rgba(0, 123, 255, 0.5);
        }
    }
    
    /* Estilos para conteúdo das mensagens */
    .chat-bubble p {
        margin: 0 0 8px 0;
    }
    
    .chat-bubble p:last-child {
        margin-bottom: 0;
    }
    
    .chat-bubble h1,
    .chat-bubble h2,
    .chat-bubble h3 {
        margin: 12px 0 8px 0;
        font-weight: 600;
        color: inherit;
    }
    
    .ai-bubble h1,
    .ai-bubble h2,
    .ai-bubble h3 {
        color: #2c3e50;
    }
    
    /* Código inline */
    .chat-bubble code {
        background-color: rgba(0,0,0,0.1);
        padding: 3px 6px;
        border-radius: 4px;
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        font-size: 0.9em;
        border: 1px solid rgba(0,0,0,0.1);
    }
    
    .human-bubble code {
        background-color: rgba(255,255,255,0.2);
        color: #f8f9fa;
        border-color: rgba(255,255,255,0.2);
    }
    
    /* Blocos de código */
    .chat-bubble pre {
        background-color: rgba(0,0,0,0.05);
        padding: 12px;
        border-radius: 6px;
        overflow-x: auto;
        margin: 12px 0;
        border: 1px solid rgba(0,0,0,0.1);
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        font-size: 0.9em;
    }
    
    .human-bubble pre {
        background-color: rgba(255,255,255,0.15);
        border-color: rgba(255,255,255,0.2);
        color: #f8f9fa;
    }
    
    /* Listas */
    .chat-bubble ul,
    .chat-bubble ol {
        padding-left: 20px;
        margin: 10px 0;
    }
    
    .chat-bubble li {
        margin-bottom: 5px;
        line-height: 1.4;
    }
    
    /* Links */
    .chat-bubble a {
        color: #007bff;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.2s;
    }
    
    .chat-bubble a:hover {
        color: #0056b3;
        text-decoration: underline;
    }
    
    .human-bubble a {
        color: #87ceeb;
    }
    
    .human-bubble a:hover {
        color: #b0e0e6;
    }
    
    /* Animações */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% { 
            opacity: 0.4; 
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
    
    /* Indicadores de status */
    .status-indicator {
        position: absolute;
        top: -5px;
        right: -5px;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: #28a745;
        border: 2px solid #fff;
        animation: pulse 2s infinite;
    }
    
    /* Timestamps */
    .timestamp {
        font-size: 0.7em;
        opacity: 0.7;
        margin-top: 5px;
        text-align: right;
    }
    
    .human-bubble .timestamp {
        color: rgba(255,255,255,0.8);
    }
    
    .ai-bubble .timestamp {
        color: #6c757d;
    }
    
    .waiting-bubble .timestamp {
        color: #856404;
    }
    
    /* Container para GIF inline */
    .inline-gif-container {
        display: flex;
        align-items: center;
        gap: 15px;
        margin: 10px 0;
    }
    
    .inline-gif {
        width: 32px;
        height: 32px;
        border-radius: 8px;
        border: 2px solid rgba(255,255,255,0.5);
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    /* Scrollbar personalizada */
    .chat-messages-container::-webkit-scrollbar {
        width: 8px;
    }
    
    .chat-messages-container::-webkit-scrollbar-track {
        background: rgba(241, 241, 241, 0.5);
        border-radius: 10px;
    }
    
    .chat-messages-container::-webkit-scrollbar-thumb {
        background: rgba(193, 193, 193, 0.8);
        border-radius: 10px;
        transition: background 0.2s;
    }
    
    .chat-messages-container::-webkit-scrollbar-thumb:hover {
        background: rgba(168, 168, 168, 0.9);
    }
    
    /* Loading dots animation melhorada */
    .loading-dots {
        display: inline-flex;
        gap: 3px;
        align-items: center;
        margin-left: 10px;
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
    
    @keyframes loadingDots {
        0%, 100% { 
            opacity: 0.3; 
            transform: scale(0.8) translateY(0);
        }
        50% { 
            opacity: 1; 
            transform: scale(1) translateY(-5px);
        }
    }
    
    /* Responsividade */
    @media (max-width: 768px) {
        .chat-bubble {
            max-width: 85%;
            padding: 10px 14px;
            font-size: 13px;
        }
        
        .chat-icon, .gif-icon {
            width: 32px;
            height: 32px;
        }
        
        .chat-row {
            margin: 8px 0;
        }
        
        .inline-gif {
            width: 28px;
            height: 28px;
        }
        
        .waiting-content {
            padding: 30px !important;
            max-width: 350px !important;
        }
        
        .waiting-gif-large {
            width: 100px !important;
            height: 100px !important;
        }
        
        .waiting-text {
            font-size: 16px !important;
        }
    }
    
    @media (max-width: 480px) {
        .chat-bubble {
            max-width: 90%;
            padding: 8px 12px;
            font-size: 12px;
        }
        
        .chat-icon, .gif-icon {
            width: 28px;
            height: 28px;
        }
        
        .chat-row {
            margin: 6px 0;
        }
        
        .inline-gif {
            width: 24px;
            height: 24px;
        }
        
        .waiting-content {
            padding: 25px !important;
            max-width: 300px !important;
        }
        
        .waiting-gif-large {
            width: 80px !important;
            height: 80px !important;
        }
        
        .waiting-text {
            font-size: 14px !important;
        }
    }
    
    /* Estados especiais */
    .chat-bubble.typing {
        animation: typing 1.5s infinite;
    }
    
    @keyframes typing {
        0%, 100% { opacity: 0.7; }
        50% { opacity: 1; }
    }
    
    /* Efeitos hover melhorados */
    .chat-bubble:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        transition: all 0.2s ease;
    }
    
    .waiting-bubble:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(255, 193, 7, 0.4);
    }
    
    /* Destaque para mensagens recentes */
    .chat-row:last-child .chat-bubble {
        animation: slideIn 0.4s ease-out;
    }
    
    /* Melhorias visuais para diferentes tipos de mensagem */
    .success-bubble {
        background: linear-gradient(135deg, rgba(212, 237, 218, 0.95) 0%, rgba(195, 230, 203, 0.95) 100%);
        border-left: 4px solid #28a745;
        color: #155724;
    }
    
    .error-bubble {
        background: linear-gradient(135deg, rgba(248, 215, 218, 0.95) 0%, rgba(245, 198, 203, 0.95) 100%);
        border-left: 4px solid #dc3545;
        color: #721c24;
    }
    
    .info-bubble {
        background: linear-gradient(135deg, rgba(209, 236, 241, 0.95) 0%, rgba(190, 229, 235, 0.95) 100%);
        border-left: 4px solid #17a2b8;
        color: #0c5460;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def find_image_files():
    """
    Procura pelos arquivos de imagem em diferentes localizações usando pathlib
    """
    current_dir = Path.cwd()
    
    # Possíveis nomes de arquivo
    robot_names = ["template_robot.png", "robot.png", "ai.png", "bot.png"]
    user_names = ["template_user.png", "user.png", "human.png", "person.png"]
    background_names = ["chat_robot_background.png", "background.png", "bg.png"]
    
    # Possíveis diretórios
    search_dirs = [
        current_dir,
        current_dir / "images",
        current_dir / "img", 
        current_dir / "assets",
        current_dir / "static",
        current_dir / "icons"
    ]
    
    robot_path = None
    user_path = None
    background_path = None
    
    # Procurar imagem do robô
    for directory in search_dirs:
        if not directory.exists():
            continue
        for name in robot_names:
            path = directory / name
            if path.exists() and path.is_file():
                robot_path = str(path)
                break
        if robot_path:
            break
    
    # Procurar imagem do usuário
    for directory in search_dirs:
        if not directory.exists():
            continue
        for name in user_names:
            path = directory / name
            if path.exists() and path.is_file():
                user_path = str(path)
                break
        if user_path:
            break
    
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
    
    return robot_path, user_path, background_path

def find_gif_files():
    """
    Procura pelos arquivos GIF de loading
    """
    current_dir = Path.cwd()
    
    # Possíveis diretórios para GIFs
    search_dirs = [
        current_dir / "img",
        current_dir / "images",
        current_dir / "assets",
        current_dir / "gifs",
        current_dir
    ]
    
    loading_gif_path = None
    waiting_gif_path = None
    
    # Procurar GIFs específicos
    for directory in search_dirs:
        if not directory.exists():
            continue
            
        # Loading screen GIF
        loading_path = directory / "loading_screen.gif"
        if loading_path.exists() and not loading_gif_path:
            loading_gif_path = str(loading_path)
        
        # Waiting GIF
        waiting_path = directory / "waiting.gif"
        if waiting_path.exists() and not waiting_gif_path:
            waiting_gif_path = str(waiting_path)
    
    return loading_gif_path, waiting_gif_path

@st.cache_data
def get_base64_image(image_path):
    """Converte imagem local para base64 com tratamento de erros melhorado"""
    try:
        path = Path(image_path)
        if not path.exists():
            return None
            
        with open(path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
            # Verificar se o encoding foi bem sucedido
            if encoded:
                return encoded
            else:
                return None
    except Exception as e:
        print(f"Erro ao carregar {image_path}: {e}")
        return None

@st.cache_data
def get_base64_gif(gif_path):
    """Converte GIF para base64 especificamente"""
    try:
        path = Path(gif_path)
        if not path.exists():
            return None
            
        with open(path, "rb") as gif_file:
            encoded = base64.b64encode(gif_file.read()).decode()
            if encoded:
                return f"data:image/gif;base64,{encoded}"
            else:
                return None
    except Exception as e:
        print(f"Erro ao carregar GIF {gif_path}: {e}")
        return None

# Imagens padrão em SVG otimizadas
DEFAULT_ROBOT_SVG = """data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Cdefs%3E%3ClinearGradient id='robotGrad' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' style='stop-color:%23007bff;stop-opacity:1' /%3E%3Cstop offset='100%25' style='stop-color:%230056b3;stop-opacity:1' /%3E%3C/linearGradient%3E%3C/defs%3E%3Ccircle cx='50' cy='50' r='45' fill='url(%23robotGrad)'/%3E%3Ccircle cx='35' cy='40' r='8' fill='white'/%3E%3Ccircle cx='65' cy='40' r='8' fill='white'/%3E%3Ccircle cx='35' cy='40' r='3' fill='%23333'/%3E%3Ccircle cx='65' cy='40' r='3' fill='%23333'/%3E%3Crect x='40' y='60' width='20' height='8' rx='4' fill='white'/%3E%3C/svg%3E"""

DEFAULT_USER_SVG = """data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Cdefs%3E%3ClinearGradient id='userGrad' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' style='stop-color:%236c757d;stop-opacity:1' /%3E%3Cstop offset='100%25' style='stop-color:%23495057;stop-opacity:1' /%3E%3C/linearGradient%3E%3C/defs%3E%3Ccircle cx='50' cy='50' r='45' fill='url(%23userGrad)'/%3E%3Ccircle cx='50' cy='35' r='15' fill='white'/%3E%3Cpath d='M25 75 Q25 60 50 60 Q75 60 75 75 Z' fill='white'/%3E%3C/svg%3E"""

# SVG animado para loading quando não há GIF
DEFAULT_LOADING_SVG = """data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Cdefs%3E%3ClinearGradient id='loadGrad' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' style='stop-color:%23ffc107;stop-opacity:1' /%3E%3Cstop offset='100%25' style='stop-color:%23ff8c00;stop-opacity:1' /%3E%3C/linearGradient%3E%3C/defs%3E%3Ccircle cx='50' cy='50' r='45' fill='url(%23loadGrad)'/%3E%3Cg%3E%3CanimateTransform attributeName='transform' attributeType='XML' type='rotate' from='0 50 50' to='360 50 50' dur='2s' repeatCount='indefinite'/%3E%3Ccircle cx='50' cy='25' r='8' fill='white'/%3E%3Ccircle cx='75' cy='50' r='6' fill='white'/%3E%3Ccircle cx='50' cy='75' r='4' fill='white'/%3E%3C/g%3E%3C/svg%3E"""

# Inicializar templates
def initialize_templates():
    """Inicializa os templates com as imagens e GIFs encontrados"""
    global ai_template, human_template, waiting_template, background_css
    
    # Tentar encontrar as imagens locais
    robot_path, user_path, background_path = find_image_files()
    loading_gif_path, waiting_gif_path = find_gif_files()
    
    print(f"🔍 Procurando recursos:")
    print(f"  Robot PNG: {'✅ ' + robot_path if robot_path else '❌ não encontrado'}")
    print(f"  User PNG: {'✅ ' + user_path if user_path else '❌ não encontrado'}")
    print(f"  Background PNG: {'✅ ' + background_path if background_path else '❌ não encontrado'}")
    print(f"  Loading GIF: {'✅ ' + loading_gif_path if loading_gif_path else '❌ não encontrado'}")
    print(f"  Waiting GIF: {'✅ ' + waiting_gif_path if waiting_gif_path else '❌ não encontrado'}")

    # Carregar imagens (local ou padrão)
    robot_img_src = DEFAULT_ROBOT_SVG
    user_img_src = DEFAULT_USER_SVG
    waiting_gif_src = DEFAULT_LOADING_SVG
    background_b64 = ""
    
    if robot_path:
        robot_b64 = get_base64_image(robot_path)
        if robot_b64:
            robot_img_src = f"data:image/png;base64,{robot_b64}"
            print("✅ Imagem do robô carregada com sucesso!")
        else:
            print("⚠️ Falha ao carregar imagem do robô, usando SVG padrão")
    
    if user_path:
        user_b64 = get_base64_image(user_path)
        if user_b64:
            user_img_src = f"data:image/png;base64,{user_b64}"
            print("✅ Imagem do usuário carregada com sucesso!")
        else:
            print("⚠️ Falha ao carregar imagem do usuário, usando SVG padrão")
    
    if background_path:
        background_b64 = get_base64_image(background_path)
        if background_b64:
            print("✅ Imagem de fundo carregada com sucesso!")
        else:
            print("⚠️ Falha ao carregar imagem de fundo")
    
    if waiting_gif_path:
        waiting_gif_b64 = get_base64_gif(waiting_gif_path)
        if waiting_gif_b64:
            waiting_gif_src = waiting_gif_b64
            print("✅ GIF de espera carregado com sucesso!")
        else:
            print("⚠️ Falha ao carregar GIF de espera, usando SVG animado padrão")

    # Aplicar fundo personalizado no CSS
    if background_b64:
        # Substituir placeholder do fundo no CSS
        css_with_background = load_css.__code__.co_consts[1].replace("{{BACKGROUND_IMAGE}}", background_b64)
        st.markdown(f"<style>{css_with_background}</style>", unsafe_allow_html=True)

    # Criar templates limpos e otimizados
    ai_template = f'''
    <div class="chat-row">
        <img class="chat-icon" src="{robot_img_src}" alt="AI" loading="lazy">
        <div class="chat-bubble ai-bubble">
            {{{{MSG}}}}
            <div class="timestamp">Agente RAG</div>
        </div>
    </div>
    '''

    human_template = f'''
    <div class="chat-row row-reverse">
        <img class="chat-icon" src="{user_img_src}" alt="Usuário" loading="lazy">
        <div class="chat-bubble human-bubble">
            {{{{MSG}}}}
            <div class="timestamp">Você</div>
        </div>
    </div>
    '''
    
    # Template de espera normal (inline)
    waiting_template = f'''
    <div class="chat-row">
        <img class="gif-icon" src="{waiting_gif_src}" alt="Processando..." loading="lazy">
        <div class="chat-bubble ai-bubble waiting-bubble">
            <div class="inline-gif-container">
                <img class="inline-gif" src="{waiting_gif_src}" alt="Carregando..." loading="lazy">
                <div>
                    <div style="font-weight: 500; margin-bottom: 5px;">🤔 Processando sua pergunta...</div>
                    <div style="font-size: 12px; opacity: 0.8;">O agente está analisando e preparando a resposta</div>
                </div>
            </div>
            <div class="timestamp">Agente RAG</div>
        </div>
    </div>
    '''
    
    print("🎨 Templates inicializados com sucesso!")

def create_centralized_waiting_overlay():
    """
    Cria overlay centralizado com GIF de espera
    """
    _, waiting_gif_path = find_gif_files()
    
    if waiting_gif_path:
        waiting_gif_b64 = get_base64_gif(waiting_gif_path)
        if waiting_gif_b64:
            return f'''
            <div class="waiting-overlay" id="waiting-overlay">
                <div class="waiting-content">
                    <img class="waiting-gif-large" src="{waiting_gif_b64}" alt="Processando..." loading="eager">
                    <div class="waiting-text">🤔 Processando sua pergunta...</div>
                    <div class="waiting-subtitle">O agente está analisando os dados e preparando uma resposta personalizada para você.</div>
                </div>
            </div>
            '''
    
    # Fallback com SVG animado
    return '''
    <div class="waiting-overlay" id="waiting-overlay">
        <div class="waiting-content">
            <div style="
                width: 120px;
                height: 120px;
                margin: 0 auto 20px auto;
                border-radius: 50%;
                border: 4px solid #007bff;
                display: flex;
                align-items: center;
                justify-content: center;
                animation: pulseGlow 2s infinite;
            ">
                <div style="
                    width: 80px;
                    height: 80px;
                    border: 4px solid rgba(0, 123, 255, 0.3);
                    border-top: 4px solid #007bff;
                    border-radius: 50%;
                    animation: spin 1s linear infinite;
                "></div>
            </div>
            <div class="waiting-text">🤔 Processando sua pergunta...</div>
            <div class="waiting-subtitle">O agente está analisando os dados e preparando uma resposta personalizada para você.</div>
        </div>
    </div>
    '''

def show_centralized_waiting():
    """
    Mostra o GIF de espera centralizado na tela
    """
    waiting_overlay = create_centralized_waiting_overlay()
    return st.markdown(waiting_overlay, unsafe_allow_html=True)

def hide_centralized_waiting():
    """
    Esconde o GIF de espera centralizado
    """
    hide_script = """
    <script>
    (function() {
        const overlay = document.getElementById('waiting-overlay');
        if (overlay) {
            overlay.style.animation = 'fadeOut 0.3s ease-out forwards';
            setTimeout(() => {
                if (overlay.parentNode) {
                    overlay.parentNode.removeChild(overlay);
                }
            }, 300);
        }
    })();
    </script>
    """
    return st.markdown(hide_script, unsafe_allow_html=True)

# Templates para outros tipos de mensagem
def create_system_template():
    return '''
    <div class="chat-row">
        <img class="chat-icon" src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Cdefs%3E%3ClinearGradient id='sysGrad' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' style='stop-color:%2317a2b8;stop-opacity:1' /%3E%3Cstop offset='100%25' style='stop-color:%23138496;stop-opacity:1' /%3E%3C/linearGradient%3E%3C/defs%3E%3Ccircle cx='50' cy='50' r='45' fill='url(%23sysGrad)'/%3E%3Ctext x='50' y='60' font-family='Arial' font-size='40' fill='white' text-anchor='middle'%3Ei%3C/text%3E%3C/svg%3E" alt="Sistema" loading="lazy">
        <div class="chat-bubble info-bubble">
            {{MSG}}
            <div class="timestamp">Sistema</div>
        </div>
    </div>
    '''

def create_error_template():
    return '''
    <div class="chat-row">
        <img class="chat-icon" src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Cdefs%3E%3ClinearGradient id='errGrad' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' style='stop-color:%23dc3545;stop-opacity:1' /%3E%3Cstop offset='100%25' style='stop-color:%23c82333;stop-opacity:1' /%3E%3C/linearGradient%3E%3C/defs%3E%3Ccircle cx='50' cy='50' r='45' fill='url(%23errGrad)'/%3E%3Ctext x='50' y='65' font-family='Arial' font-size='50' fill='white' text-anchor='middle'%3E!%3C/text%3E%3C/svg%3E" alt="Erro" loading="lazy">
        <div class="chat-bubble error-bubble">
            {{MSG}}
            <div class="timestamp">Erro</div>
        </div>
    </div>
    '''

def create_success_template():
    return '''
    <div class="chat-row">
        <img class="chat-icon" src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Cdefs%3E%3ClinearGradient id='sucGrad' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' style='stop-color:%2328a745;stop-opacity:1' /%3E%3Cstop offset='100%25' style='stop-color:%2320c997;stop-opacity:1' /%3E%3C/linearGradient%3E%3C/defs%3E%3Ccircle cx='50' cy='50' r='45' fill='url(%23sucGrad)'/%3E%3Cpath d='M25 50 L45 65 L75 35' stroke='white' stroke-width='8' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E" alt="Sucesso" loading="lazy">
        <div class="chat-bubble success-bubble">
            {{MSG}}
            <div class="timestamp">Sucesso</div>
        </div>
    </div>
    '''

def create_waiting_message():
    """
    Cria mensagem de espera dinâmica baseada nos GIFs disponíveis
    """
    if 'waiting_template' not in globals():
        initialize_templates()
    
    return waiting_template

def get_loading_screen_html():
    """
    Retorna HTML da tela de loading inicial otimizado
    """
    loading_gif_path, _ = find_gif_files()
    
    if loading_gif_path:
        loading_gif_b64 = get_base64_gif(loading_gif_path)
        if loading_gif_b64:
            return f"""
            <div id="loading-screen" style="
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                z-index: 9999;
                color: white;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            ">
                <img src="{loading_gif_b64}" alt="Carregando..." style="
                    max-width: 300px;
                    max-height: 300px;
                    border-radius: 20px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                    margin-bottom: 30px;
                    border: 3px solid rgba(255,255,255,0.2);
                " loading="eager">
                <h2 style="
                    margin: 0 0 15px 0;
                    font-size: 28px;
                    font-weight: 300;
                    text-shadow: 0 2px 10px rgba(0,0,0,0.3);
                    animation: fadeInOut 2s infinite;
                ">Inicializando Agente RAG</h2>
                <p style="
                    margin: 0;
                    font-size: 16px;
                    opacity: 0.8;
                    text-align: center;
                    max-width: 400px;
                    line-height: 1.5;
                ">Preparando a inteligência artificial para conversar com você...</p>
                
               
            </div>
            
            <style>
            @keyframes fadeInOut {{
                0%, 100% {{ opacity: 0.7; }}
                50% {{ opacity: 1; }}
            }}
            
            @keyframes progressGlow {{
                0% {{ background-position: 0% 50%; }}
                100% {{ background-position: 100% 50%; }}
            }}
            
            @keyframes fadeOut {{
                from {{ opacity: 1; transform: scale(1); }}
                to {{ opacity: 0; transform: scale(0.9); }}
            }}
            </style>"""
    
    # Fallback com SVG animado otimizado
    return """
    <div id="loading-screen" style="
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        z-index: 9999;
        color: white;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    ">
        <!-- SVG animado como fallback -->
        <div style="
            width: 120px;
            height: 120px;
            margin-bottom: 30px;
        ">
            <svg viewBox="0 0 100 100" style="width: 100%; height: 100%;">
                <defs>
                    <linearGradient id="loadGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" style="stop-color:#ffc107;stop-opacity:1" />
                        <stop offset="100%" style="stop-color:#ff8c00;stop-opacity:1" />
                    </linearGradient>
                </defs>
                <circle cx="50" cy="50" r="45" fill="url(#loadGrad)" opacity="0.3"/>
                <g>
                    <animateTransform attributeName="transform" attributeType="XML" type="rotate" from="0 50 50" to="360 50 50" dur="2s" repeatCount="indefinite"/>
                    <circle cx="50" cy="15" r="8" fill="white"/>
                    <circle cx="85" cy="50" r="6" fill="white"/>
                    <circle cx="50" cy="85" r="4" fill="white"/>
                    <circle cx="15" cy="50" r="6" fill="white"/>
                </g>
            </svg>
        </div>
        <h2 style="margin: 0 0 15px 0; font-size: 28px; font-weight: 300;">Inicializando Agente RAG</h2>
        <p style="margin: 0; font-size: 16px; opacity: 0.8;">Preparando a inteligência artificial...</p>
    </div>
    
    <style>
    @keyframes fadeOut {
        from { opacity: 1; transform: scale(1); }
        to { opacity: 0; transform: scale(0.9); }
    }
    </style>
    """

# Função auxiliar para escapar HTML de forma segura
def escape_html(text):
    """
    Escapa caracteres especiais HTML preservando quebras de linha.
    """
    if not text:
        return ""
    return html.escape(str(text)).replace("\n", "<br>")

# Função auxiliar para processar markdown simples
def simple_markdown(text):
    """
    Processa markdown básico em HTML de forma segura.
    """
    if not text:
        return ""
    
    text = str(text)
    
    # Primeiro escapar HTML para segurança
    text = html.escape(text)
    
    # Processar markdown
    # Negrito
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'__(.*?)__', r'<strong>\1</strong>', text)
    
    # Itálico
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    text = re.sub(r'_(.*?)_', r'<em>\1</em>', text)
    
    # Código inline
    text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
    
    # Links (processar com cuidado)
    text = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'<a href="\2" target="_blank" rel="noopener noreferrer">\1</a>', text)
    
    # Quebras de linha
    text = text.replace('\n', '<br>')
    
    return text

# Função auxiliar para criar mensagens formatadas
def create_message(message_type, content, timestamp=None):
    """
    Cria uma mensagem formatada com base no tipo.
    
    Args:
        message_type (str): Tipo da mensagem ('ai', 'human', 'system', 'error', 'loading', 'success', 'waiting')
        content (str): Conteúdo da mensagem
        timestamp (str, optional): Timestamp personalizado
    
    Returns:
        str: HTML formatado da mensagem
    """
    if not content:
        content = ""
    
    # Processar markdown e escapar HTML
    processed_content = simple_markdown(content)
    
    # Garantir que os templates estão inicializados
    if 'ai_template' not in globals():
        initialize_templates()
    
    templates = {
        'ai': ai_template,
        'human': human_template,
        'system': create_system_template(),
        'error': create_error_template(),
        'success': create_success_template(),
        'waiting': create_waiting_message()
    }
    
    template = templates.get(message_type, ai_template)
    
    # Substituir placeholder de forma segura
    return template.replace("{{MSG}}", processed_content)

# Função para renderizar múltiplas mensagens
def render_messages(messages):
    """
    Renderiza uma lista de mensagens de forma otimizada.
    
    Args:
        messages (list): Lista de objetos Message
    
    Returns:
        str: HTML formatado de todas as mensagens
    """
    if not messages:
        return ""
    
    html_messages = []
    
    for message in messages:
        try:
            if hasattr(message, 'origin') and hasattr(message, 'message'):
                html_messages.append(create_message(message.origin, message.message))
            elif isinstance(message, dict):
                msg_type = message.get('origin', message.get('type', 'system'))
                msg_content = message.get('message', message.get('content', str(message)))
                html_messages.append(create_message(msg_type, msg_content))
            else:
                # Fallback para mensagens em formato diferente
                html_messages.append(create_message('system', str(message)))
        except Exception as e:
            print(f"Erro ao renderizar mensagem: {e}")
            html_messages.append(create_message('error', f"Erro ao exibir mensagem: {str(e)}"))
    
    return "\n".join(html_messages)

# Função para criar container de mensagens
def create_chat_container(messages, container_height="70vh"):
    """
    Cria um container completo para as mensagens do chat.
    
    Args:
        messages (list): Lista de mensagens
        container_height (str): Altura do container
    
    Returns:
        str: HTML completo do container
    """
    messages_html = render_messages(messages)
    
    return f"""
    <div class="chat-messages-container" style="max-height: {container_height};">
        {messages_html if messages_html else '<div style="text-align: center; padding: 50px; color: #6c757d;"><h3>🌟 Bem-vindo ao Chat RAG!</h3><p>Faça sua primeira pergunta para começar a conversa.</p></div>'}
    </div>
    """

# Função para instruções de uso
def print_usage_instructions():
    """
    Imprime instruções sobre como usar imagens e GIFs personalizados
    """
    print("\n" + "="*60)
    print("📋 INSTRUÇÕES PARA USAR RECURSOS PERSONALIZADOS")
    print("="*60)
    print("Para usar suas próprias imagens e GIFs:")
    print("\n🖼️ IMAGENS ESTÁTICAS:")
    print("   • template_robot.png (ícone do robô/AI)")
    print("   • template_user.png (ícone do usuário)")
    print("   • chat_robot_background.png (fundo da aplicação)")
    
    print("\n🎬 GIFS DE LOADING:")
    print("   • loading_screen.gif (tela inicial de carregamento)")
    print("   • waiting.gif (aguardando resposta do bot)")
    
    print("\n📂 ORGANIZAÇÃO DE PASTAS:")
    print("   • img/ (recomendado para GIFs)")
    print("   • images/ | assets/ | icons/ (para imagens)")
    
    print("\n✅ ESPECIFICAÇÕES:")
    print("   • Formato GIF: Animado, até 5MB")
    print("   • Formato PNG/JPG: 36x36px ou maior")
    print("   • Fundo: Qualquer resolução, será redimensionado")
    print("   • Fundo transparente (recomendado para ícones)")
    
    print("\n🔍 O sistema procura automaticamente em:")
    print("   • Diretório atual")
    print("   • img/ | images/ | assets/ | static/ | icons/")
    
    print("\n💡 NOVOS RECURSOS:")
    print("   • 🎯 GIF de espera centralizado na tela")
    print("   • 🖼️ Fundo personalizado fixo")
    print("   • ✨ Fallback automático para SVG se não encontrar")
    print("   • 🎨 Animações CSS integradas")
    print("   • 📱 Responsivo para mobile")
    print("   • 🚀 Cache otimizado")
    print("   • ⚡ Correção automática para quadrado branco")
    print("\n" + "="*60)

# Função de limpeza para elementos problemáticos
def clean_streamlit_elements():
    """
    Retorna JavaScript para limpar elementos que causam quadrado branco
    """
    return """
    <script>
    (function() {
        'use strict';
        
        function cleanupStreamlitElements() {
            // Remover iframes vazios
            const emptyIframes = document.querySelectorAll('iframe[width="0"], iframe[height="0"]');
            emptyIframes.forEach(iframe => {
                if (iframe.parentNode) {
                    iframe.style.display = 'none';
                    iframe.style.visibility = 'hidden';
                    iframe.style.position = 'absolute';
                    iframe.style.left = '-9999px';
                }
            });
            
            // Remover divs vazias problemáticas
            const emptyDivs = document.querySelectorAll('div:empty:not(.loading-dots div):not(.status-indicator):not(.chat-icon):not(.inline-gif)');
            emptyDivs.forEach(div => {
                if (div.parentNode && !div.classList.contains('loading-dots') && 
                    !div.classList.contains('status-indicator') && 
                    !div.classList.contains('chat-icon') && 
                    !div.classList.contains('inline-gif')) {
                    div.style.display = 'none';
                }
            });
            
            // Remover elementos com dimensões zero
            const zeroDimElements = document.querySelectorAll('[style*="height: 0"], [style*="width: 0"]');
            zeroDimElements.forEach(el => {
                if (!el.classList.contains('loading-dots') && 
                    !el.classList.contains('status-indicator')) {
                    el.style.display = 'none';
                }
            });
            
            // Limpar containers vazios do Streamlit
            const stContainers = document.querySelectorAll('.stApp > div[data-testid="stAppViewContainer"] > div:empty');
            stContainers.forEach(container => {
                container.style.display = 'none';
            });
        }
        
        // Executar limpeza inicial
        setTimeout(cleanupStreamlitElements, 500);
        
        // Executar limpeza periódica
        setInterval(cleanupStreamlitElements, 2000);
        
        // Observer para limpeza automática
        const observer = new MutationObserver(cleanupStreamlitElements);
        observer.observe(document.body, { childList: true, subtree: true });
        
    })();
    </script>
    """

# Inicializar templates na importação
initialize_templates()

# Executar instruções se executado diretamente
if __name__ == "__main__":
    print_usage_instructions()
    print(f"\n🧪 Testando inicialização dos templates...")
    print(f"✅ Templates inicializados com sucesso!")
    print(f"📊 AI Template: {len(ai_template)} caracteres")
    print(f"📊 Human Template: {len(human_template)} caracteres")
    print(f"📊 Waiting Template: {len(waiting_template) if 'waiting_template' in globals() else 0} caracteres")
    print(f"🔧 Sistema de limpeza ativo para prevenir quadrado branco")
    print(f"🎯 Sistema de GIF centralizado implementado")

    print(f"🖼️ Sistema de fundo personalizado implementado")
