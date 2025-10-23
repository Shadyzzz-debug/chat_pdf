import os
import streamlit as st
import numpy as np 
from PIL import Image
from PyPDF2 import PdfReader
from requests import post # Usado para llamadas directas a la API de OpenAI
import json
import platform

# --- Configuración de Modelos ---
# Utilizaremos el modelo estándar de embeddings y un modelo de chat eficiente para la QA.
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_CHAT_MODEL = "gpt-4o-mini" 

# --- Funciones de Utilidad (Sin LangChain) ---

def get_openai_embedding(text: str, api_key: str) -> np.ndarray:
    """Obtiene el embedding de un texto directamente desde la API de OpenAI."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": OPENAI_EMBEDDING_MODEL,
        "input": text,
    }
    # La API de OpenAI devuelve un error si se llama con una clave vacía o inválida.
    response = post("https://api.openai.com/v1/embeddings", headers=headers, json=payload)
    response.raise_for_status() # Lanza una excepción para errores HTTP (p. ej., 401 si la clave es mala)
    data = response.json()
    return np.array(data['data'][0]['embedding'], dtype=np.float32)

def calculate_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Calcula la similitud coseno entre dos vectores NumPy."""
    # La similitud coseno se calcula como el producto punto normalizado.
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

def get_llm_answer(context: str, question: str, api_key: str) -> str:
    """Obtiene la respuesta del LLM directamente desde la API de OpenAI."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    # Construir el prompt RAG con el contexto
    prompt = (
        "Eres un asistente experto que responde preguntas basándose ÚNICAMENTE en el contexto proporcionado. "
        "Si la respuesta no se encuentra en el contexto, indica de manera concisa que la información no está disponible en el documento. "
        f"Contexto:\n---\n{context}\n---\nPregunta: {question}"
    )
    
    payload = {
        "model": OPENAI_CHAT_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0 # Asegura respuestas deterministas basadas en el contexto
    }
    
    response = post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status() # Lanza una excepción si la llamada falla
    data = response.json()
    
    # Manejo de la respuesta
    if 'choices' in data and len(data['choices']) > 0:
        return data['choices'][0]['message']['content']
    return "Error: No se recibió una respuesta válida del Oráculo."

# Función de división de texto (simulando CharacterTextSplitter)
def split_text_into_chunks(text: str, chunk_size=500, chunk_overlap=20):
    """Divide el texto en fragmentos con superposición."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks

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
st.title('📜 Generación Aumentada por Recuperación (RAG) - Versión Pura')
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
    
    *Nota: Esta versión no usa LangChain, implementando RAG con llamadas directas a la API de OpenAI y NumPy.*
    """)

# Get API key from user
ke = st.text_input('Ingresa tu Clave de Esencia Vital (OpenAI API Key)', type="password")
if ke:
    # No es necesario os.environ['OPENAI_API_KEY'] ya que usamos 'ke' directamente en las funciones.
    pass
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
        
        # --- Lógica de Almacenamiento y Recuperación (Vectorización en session_state) ---
        # Si la base de conocimiento no existe o el PDF ha cambiado, la creamos
        if 'chunk_embeddings' not in st.session_state or st.session_state.get('pdf_name') != pdf.name:
            
            # 1. Dividir el texto en fragmentos (Ritos de Fragmentación)
            with st.spinner("Fragmentando el Códice para la Memoria Arcana..."):
                # Usamos la función de división simple en lugar de LangChain's CharacterTextSplitter
                chunks = split_text_into_chunks(text, chunk_size=500, chunk_overlap=20)
            
            st.success(f"Códice Fragmentado en **{len(chunks)}** secciones de Memoria")

            # 2. Crear embeddings (El Altar de la Memoria)
            with st.spinner("Inscribiendo Fragmentos en el Altar de la Memoria (Vectorización)..."):
                
                chunk_embeddings = []
                for i, chunk in enumerate(chunks):
                    if i % 20 == 0: # Muestra progreso cada 20 chunks para evitar spam
                        st.caption(f"Vectorizando fragmento {i+1}/{len(chunks)}...")
                    
                    embedding = get_openai_embedding(chunk, ke)
                    chunk_embeddings.append(embedding)

                # Almacenar chunks y embeddings en la sesión
                st.session_state.chunk_texts = chunks
                st.session_state.chunk_embeddings = np.array(chunk_embeddings)
                st.session_state.pdf_name = pdf.name
        
        else:
            # Si ya está en la sesión, recuperamos
            chunks = st.session_state.chunk_texts
            chunk_embeddings = st.session_state.chunk_embeddings


        st.success("Altar de la Memoria (Fragmentos Vectorizados) Erigido exitosamente.")

        # Interfaz de pregunta del usuario
        st.subheader("Escribe el Interrogatorio sobre el Papiro")
        user_question = st.text_area(" ", placeholder="¿Qué verdad oculta este pergamino? Escribe tu pregunta aquí...", key="user_q")
        
        # Procesa la pregunta cuando se envía
        if user_question:
            with st.spinner("Invocando el Oráculo Aumentado y Buscando Similitud..."):
                
                # 1. Obtener embedding de la pregunta
                query_embedding = get_openai_embedding(user_question, ke)
                
                # 2. Calcular similitud (cosine similarity)
                similarities = [
                    calculate_cosine_similarity(query_embedding, emb)
                    for emb in chunk_embeddings
                ]
                
                # 3. Obtener los índices de los 4 fragmentos más similares (Recuperación)
                top_k_indices = np.argsort(similarities)[::-1][:4] # Top 4 chunks
                
                # 4. Construir el contexto
                retrieved_chunks = [chunks[i] for i in top_k_indices]
                context = "\n\n---\n\n".join(retrieved_chunks)
                
                # 5. Llamar al LLM con el contexto (Generación)
                response = get_llm_answer(context, user_question, ke)
            
            # Mostrar la respuesta
            st.markdown("### 🗣️ Respuesta del Oráculo:")
            st.markdown(response)
                
    except Exception as e:
        # Captura errores de la API (p. ej., clave inválida, límite de rate) o de procesamiento de PDF
        st.error(f"Error Crítico al procesar el Papiro o al llamar a la API: {str(e)}")
        
elif pdf is not None and not ke:
    st.warning("El Papiro está cargado, pero la Clave de Esencia Vital es necesaria.")
else:
    st.info("Por favor, carga un Papiro Digital (PDF) y proporciona la Clave de Esencia Vital para comenzar la Interrogación.")

# Información adicional y pie de página
st.markdown("---")
st.caption("""
**Notas del Erudito**: Este Códice utiliza los principios de RAG (Generación Aumentada por Recuperación) con llamadas directas a la API de OpenAI y NumPy para la vectorización.
""")
