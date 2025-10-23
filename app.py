import os
import streamlit as st
import numpy as np 
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# --- CSS GÓTICO (Paleta Arcano-Escarlata) ---
gothic_css_variant = """
<style>
/* Paleta base: Fondo #111111, Texto #E0E0E0 (Pergamino ligero), Acento #5A4832 (Bronce/Metal), Sangre #A50000 */
.stApp {
    background-color: #111111;
    color: #E0E0E0;
    font-family: 'Georgia', serif;
}

/* Título Principal (h1) */
h1 {
    color: #A50000; /* Rojo sangre */
    text-shadow: 3px 3px 8px #000000;
    font-size: 3.2em; 
    border-bottom: 5px solid #5A4832; /* Borde Bronce */
    padding-bottom: 10px;
    margin-bottom: 30px;
    text-align: center;
    letter-spacing: 2px;
}

/* Subtítulos (h2, h3): Énfasis en el bronce */
h2, h3 {
    color: #C0C0C0; /* Plata/gris claro */
    border-left: 5px solid #5A4832;
    padding-left: 10px;
    margin-top: 25px;
}

/* Input y Camera (El Papiro de Inscripción) */
div[data-testid="stTextInput"], div[data-testid="stTextarea"], .stFileUploader, .stCameraInput {
    background-color: #1A1A1A;
    border: 1px solid #5A4832;
    padding: 10px;
    border-radius: 5px;
    color: #F5F5DC;
}

/* Dataframe (No hay en este script, pero por consistencia) */
div[data-testid="stDataFrame"] table {
    background-color: #1A1A1A;
    border: 1px solid #5A4832;
    color: #E0E0E0;
}
div[data-testid="stDataFrame"] thead tr th {
    background-color: #2A2A2A !important;
    color: #A50000 !important;
}

/* Texto de Alertas (Revelaciones) */
.stSuccess { background-color: #20251B; color: #F5F5DC; border-left: 5px solid #5A4832; }
.stInfo { background-color: #1A1A25; color: #F5F5DC; border-left: 5px solid #5A4832; }
.stWarning { background-color: #352A1A; color: #F5F5DC; border-left: 5px solid #A50000; }

/* Streamlit Sidebar Background */
.css-1d3w5rq {
    background-color: #202020;
}
</style>
"""
st.markdown(gothic_css_variant, unsafe_allow_html=True)

# Configuración de página Streamlit
st.set_page_config(
    page_title="El Códice del Oráculo Documental",
    page_icon="📜",
    layout="wide"
)

# App title and presentation
st.title('📜 Generación Aumentada por Recuperación (RAG)')
st.write(f"Versión del Scriptorium (Python): **{platform.python_version()}**")

# Load and display image 
try:
    # Nota: Este archivo ('Chat_pdf.png') debe estar presente en el entorno de ejecución.
    image = Image.open('Chat_pdf.png')
    st.image(image, width=350, caption="El Sello de la Sabiduría")
except Exception as e:
    st.warning(f"No se pudo cargar el Sello de la Sabiduría ('Chat_pdf.png'). {e}")

# Sidebar information
with st.sidebar:
    st.subheader("El Escriba del Conocimiento Oculto")
    st.markdown("""
    Este Agente te permitirá interrogar el conocimiento cifrado en el **Papiro Digital (PDF)** cargado.
    Requiere la **Esencia Vital de OpenAI (API Key)** para invocar sus poderes.
    """)

# Get API key from user
ke = st.text_input('Ingresa tu Clave de Esencia Vital (OpenAI API Key)', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("⚠️ Introduce tu Clave de Esencia Vital (OpenAI API Key) para activar el Códice.")

# PDF uploader
pdf = st.file_uploader("Carga el Papiro Digital (Archivo PDF)", type="pdf")

# Process the PDF if uploaded
if pdf is not None and ke:
    try:
        # Extraer el texto del Papiro Digital
        with st.spinner("Extracto de Runas y Símbolos del Papiro..."):
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text
                
            if not text:
                 st.error("El Papiro Digital no contiene texto legible (o está vacío).")
                 st.stop()
        
        st.info(f"Papiro Descifrado: **{len(text)}** caracteres")
        
        # --- Lógica de Almacenamiento y Recuperación (FAISS en session_state) ---
        # Si la base de conocimiento no existe o el PDF ha cambiado, la creamos
        if 'knowledge_base' not in st.session_state or st.session_state.get('pdf_name') != pdf.name:
            
            # 1. Dividir el texto en fragmentos (Ritos de Fragmentación)
            with st.spinner("Fragmentando el Códice para la Memoria Arcana..."):
                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=500,
                    chunk_overlap=20,
                    length_function=len
                )
                chunks = text_splitter.split_text(text)
            
            st.success(f"Códice Fragmentado en **{len(chunks)}** secciones de Memoria")

            # 2. Crear embeddings y base de conocimiento (El Altar de la Memoria)
            with st.spinner("Inscribiendo Fragmentos en el Altar de la Memoria (FAISS)..."):
                embeddings = OpenAIEmbeddings()
                knowledge_base = FAISS.from_texts(chunks, embeddings)
                st.session_state.knowledge_base = knowledge_base
                st.session_state.pdf_name = pdf.name
        
        else:
            # Si ya está en la sesión, la recuperamos
            knowledge_base = st.session_state.knowledge_base

        st.success("Altar de la Memoria (FAISS) Erigido exitosamente.")

        # Interfaz de pregunta del usuario
        st.subheader("Escribe el Interrogatorio sobre el Papiro")
        user_question = st.text_area(" ", placeholder="¿Qué verdad oculta este pergamino? Escribe tu pregunta aquí...", key="user_q")
        
        # Procesa la pregunta cuando se envía
        if user_question:
            with st.spinner("Invocando el Oráculo Aumentado..."):
                # Búsqueda de documentos
                docs = knowledge_base.similarity_search(user_question)
                
                # Inicializar el modelo LLM
                llm = OpenAI(temperature=0, model_name="gpt-4o")
                
                # Cargar la cadena de Preguntas y Respuestas
                chain = load_qa_chain(llm, chain_type="stuff")
                
                # Ejecutar la cadena
                response = chain.run(input_documents=docs, question=user_question)
            
            # Mostrar la respuesta
            st.markdown("### 🗣️ Respuesta del Oráculo:")
            st.markdown(response)
                
    except Exception as e:
        st.error(f"Error Crítico al procesar el Papiro: {str(e)}")
        # Mostrar el error detallado para depuración
        import traceback
        st.error(traceback.format_exc())
        
elif pdf is not None and not ke:
    st.warning("El Papiro está cargado, pero la Clave de Esencia Vital es necesaria.")
else:
    st.info("Por favor, carga un Papiro Digital (PDF) y proporciona la Clave de Esencia Vital para comenzar la Interrogación.")

# Información adicional y pie de página
st.markdown("---")
st.caption("""
**Notas del Erudito**: Este Códice utiliza los principios de RAG (Generación Aumentada por Recuperación) con LangChain y OpenAI.
""")

