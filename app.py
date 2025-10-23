import streamlit as st
import numpy as np 
from PIL import Image
import json
import platform
import time # Añadir import para pausas en la vectorización

# Configuración de página Streamlit (DEBE SER LA PRIMERA LLAMADA ST.)
st.set_page_config(
    page_title="El Códice del Oráculo Documental",
    page_icon="📜",
    layout="wide"
)

# --- Configuraciones del LLM para el entorno (Gemini) ---
GEMINI_EMBEDDING_MODEL = "text-embedding-004" 
GEMINI_CHAT_MODEL = "gemini-2.5-flash-preview-09-2025" 
API_KEY = "" # Clave dejada vacía para que sea provista por el entorno de Canvas

# --- Funciones de Utilidad (Sin Librerías Externas no esenciales) ---

def safe_fetch(url, method='POST', headers=None, body=None, max_retries=3, delay=1):
    """
    Realiza llamadas a la API con reintentos y retroceso exponencial.
    Utiliza st.legacy_fetch.
    """
    if headers is None:
        headers = {'Content-Type': 'application/json'}
    
    for attempt in range(max_retries):
        try:
            # Usar st.legacy_fetch (síncrona)
            response = st.legacy_fetch(url, method=method, headers=headers, body=body)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code in [429, 500, 503] and attempt < max_retries - 1:
                time.sleep(delay * (2 ** attempt))
                continue
            else:
                error_detail = response.text if response.text else f"Código de estado: {response.status_code}"
                raise Exception(f"Fallo en la llamada a la API. {error_detail}")
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay * (2 ** attempt))
                continue
            raise e
    raise Exception("Llamada a la API fallida después de múltiples reintentos.")


def get_gemini_embedding(text: str) -> np.ndarray:
    """Obtiene el embedding de un texto directamente desde la API de Gemini (síncrona)."""
    
    payload = {
        "model": GEMINI_EMBEDDING_MODEL,
        "content": {
            "parts": [{"text": text}]
        }
    }
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_EMBEDDING_MODEL}:embedContent?key={API_KEY}"
    
    response_data = safe_fetch(apiUrl, body=json.dumps(payload))
    
    if response_data and 'embedding' in response_data:
        return np.array(response_data['embedding']['values'], dtype=np.float32)
    
    raise Exception(f"Fallo al obtener embedding: {response_data.get('error', 'Error desconocido')}")


def calculate_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Calcula la similitud coseno entre dos vectores NumPy."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(vec_a, vec_b) / (norm_a * norm_b)

def get_llm_answer(context: str, question: str) -> str:
    """Obtiene la respuesta del LLM directamente desde la API de Gemini (síncrona)."""
    
    systemPrompt = (
        "Eres un asistente experto que responde preguntas basándose ÚNICAMENTE en el contexto proporcionado. "
        "Si la respuesta no se encuentra en el contexto, indica de manera concisa que la información no está disponible en el documento. "
    )
    userQuery = f"Contexto:\n---\n{context}\n---\nPregunta: {question}"

    payload = {
        "contents": [{"parts": [{"text": userQuery}]}],
        "systemInstruction": {"parts": [{"text": systemPrompt}]},
    }
    
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_CHAT_MODEL}:generateContent?key={API_KEY}"

    response_data = safe_fetch(apiUrl, body=json.dumps(payload))
    
    candidate = response_data.get('candidates', [{}])[0]
    text = candidate.get('content', {}).get('parts', [{}])[0].get('text', None)

    if text:
        return text
    return "Error: No se recibió una respuesta válida del Oráculo de Gemini."

def split_text_into_chunks(text: str, chunk_size=500, chunk_overlap=20):
    """Divide el texto en fragmentos con superposición."""
    chunks = []
    text_segments = text.split('\n')
    current_chunk = ""
    for segment in text_segments:
        if len(current_chunk) + len(segment) + 1 <= chunk_size:
            current_chunk += segment + "\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else ""
                current_chunk = overlap_text + segment + "\n"
            else:
                chunks.append(segment[:chunk_size])
                current_chunk = segment[chunk_size-chunk_overlap:]
    
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return [chunk for chunk in chunks if chunk]


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

# App title and presentation
st.title('📜 El Códice del Oráculo Documental (RAG Texto Crudo)')
st.write(f"Versión del Scriptorium (Python): **{platform.python_version()}**")

# Load and display image 
try:
    image = Image.open('xxx.avif')
    st.image(image, width=350, caption="El Sello de la Sabiduría")
except Exception as e:
    st.warning(f"No se pudo cargar el Sello de la Sabiduría ('xxx.avif'). {e}")

# Sidebar information
with st.sidebar:
    st.subheader("El Escriba del Conocimiento Oculto")
    st.markdown("""
    **Ritual de la Revelación (Modo Texto)**
    
    Este Agente requiere que **PEGUES** el texto completo del documento aquí para forjar el **Altar de la Memoria Vectorial**.
    
    *Utiliza los secretos de Gemini para la vectorización y la respuesta oracular.*
    """)

# Nuevo input: Área de texto para pegar el contenido del documento
document_text = st.text_area("✍️ Pega el Texto Completo del Papiro Digital Aquí", height=300, key="doc_text")

# Procesar el texto si ha sido pegado
if document_text:
    try:
        text = document_text
        
        if not text.strip():
             st.error("El Área de Texto no contiene contenido legible. El pergamino está en blanco.")
             st.stop()
        
        st.info(f"Papiro Descifrado: **{len(text)}** glifos (caracteres)")
        
        text_hash = hash(text)

        # --- Lógica de Almacenamiento y Recuperación (Vectorización en session_state) ---
        if 'chunk_embeddings' not in st.session_state or st.session_state.get('text_hash') != text_hash:
            
            # 1. Dividir el texto en fragmentos (Ritos de Fragmentación)
            with st.spinner("Fragmentando el Códice para la Memoria Arcana..."):
                chunks = split_text_into_chunks(text, chunk_size=500, chunk_overlap=20)
            
            st.success(f"Códice Fragmentado en **{len(chunks)}** secciones de Memoria")

            # 2. Crear embeddings (El Altar de la Memoria)
            with st.spinner("Inscribiendo Fragmentos en el Altar de la Memoria (Vectorización)..."):
                
                chunk_embeddings = []
                for i, chunk in enumerate(chunks):
                    st.caption(f"Vectorizando fragmento {i+1}/{len(chunks)}...")
                        
                    embedding = get_gemini_embedding(chunk)
                    chunk_embeddings.append(embedding)

                # Almacenar en la sesión
                st.session_state.chunk_texts = chunks
                st.session_state.chunk_embeddings = np.array(chunk_embeddings)
                st.session_state.text_hash = text_hash
        
        else:
            # Recuperar de la sesión
            chunks = st.session_state.chunk_texts
            chunk_embeddings = st.session_state.chunk_embeddings


        st.success("Altar de la Memoria (Fragmentos Vectorizados) Erigido exitosamente.")

        # Interfaz de pregunta del usuario
        st.subheader("El Gran Interrogatorio")
        user_question = st.text_area("Invoca la Verdad", placeholder="¿Qué verdad oculta este pergamino? Escribe tu pregunta aquí para invocar la respuesta del Oráculo...", key="user_q")
        
        # Procesa la pregunta cuando se envía
        if user_question:
            with st.spinner("Invocando el Oráculo Aumentado y Buscando Similitud Arcana..."):
                
                # 1. Obtener embedding de la pregunta
                query_embedding = get_gemini_embedding(user_question)
                
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
                response = get_llm_answer(context, user_question)
            
            # Mostrar la respuesta
            st.markdown("### 🗣️ Respuesta del Oráculo de Gemini:")
            st.markdown(response)
                
    except Exception as e:
        st.error(f"Error Crítico al procesar el Papiro o al llamar a la API: {str(e)}. Verifica la configuración del Oráculo.")
        
else:
    st.info("Introduce el Texto en el pergamino superior para Despertar el Agente.")

# Información adicional y pie de página
st.markdown("---")
st.caption("""
**Notas del Erudito**: Este Códice opera asumiendo que solo se requiere `numpy` además de las librerías estándar de Python. Se ha modificado para usar llamadas síncronas con reintentos para mayor estabilidad.
""")
