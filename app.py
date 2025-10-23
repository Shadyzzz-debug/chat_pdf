import streamlit as st
import numpy as np 
from PIL import Image
import json
import platform

# --- Configuraciones del LLM para el entorno ---
# Usamos la API de Gemini (disponible internamente) para la generación de embeddings y respuestas.
# En este entorno, 'fetch' maneja la comunicación con la API.
GEMINI_EMBEDDING_MODEL = "text-embedding-004" 
GEMINI_CHAT_MODEL = "gemini-2.5-flash-preview-09-2025" 
API_KEY = "" # Clave dejada vacía para que sea provista por el entorno de Canvas

# --- Funciones de Utilidad (Sin Librerías Externas no esenciales) ---

async def get_gemini_embedding(text: str) -> np.ndarray:
    """Obtiene el embedding de un texto directamente desde la API de Gemini."""
    
    # Payload para la API de Embeddings de Gemini
    payload = {
        "model": GEMINI_EMBEDDING_MODEL,
        "content": {
            "parts": [{"text": text}]
        }
    }
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_EMBEDDING_MODEL}:embedContent?key={API_KEY}"
    
    # Llama a la API (Asumiendo que 'fetch' está disponible en el entorno)
    response = await st.experimental_rerun_with_fetch(apiUrl, method='POST', headers={'Content-Type': 'application/json'}, body=json.dumps(payload))
    
    # Procesar la respuesta
    if response and 'embedding' in response:
        return np.array(response['embedding']['values'], dtype=np.float32)
    
    raise Exception(f"Fallo al obtener embedding: {response.get('error', 'Error desconocido')}")


def calculate_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Calcula la similitud coseno entre dos vectores NumPy."""
    # Requiere NumPy (asumido como parte del core)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(vec_a, vec_b) / (norm_a * norm_b)

async def get_llm_answer(context: str, question: str) -> str:
    """Obtiene la respuesta del LLM directamente desde la API de Gemini (Simulación de fetch)."""
    
    # Construir el prompt RAG con el contexto
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

    # Llama a la API (Simulación de fetch asíncrono)
    response = await st.experimental_rerun_with_fetch(apiUrl, method='POST', headers={'Content-Type': 'application/json'}, body=json.dumps(payload))
    
    # Manejo de la respuesta
    candidate = response.get('candidates', [{}])[0]
    text = candidate.get('content', {}).get('parts', [{}])[0].get('text', None)

    if text:
        return text
    return "Error: No se recibió una respuesta válida del Oráculo de Gemini."

# Función de división de texto
def split_text_into_chunks(text: str, chunk_size=500, chunk_overlap=20):
    """Divide el texto en fragmentos con superposición."""
    chunks = []
    # Usar una división basada en saltos de línea y luego por tamaño.
    text_segments = text.split('\n')
    
    current_chunk = ""
    for segment in text_segments:
        if len(current_chunk) + len(segment) + 1 <= chunk_size:
            current_chunk += segment + "\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # El solapamiento se gestiona simplificando el inicio del nuevo chunk
                # Podríamos buscar el solapamiento más preciso, pero para cero dependencias, simplificamos.
                overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else ""
                current_chunk = overlap_text + segment + "\n"
            else:
                # Caso extremo: un solo segmento es más grande que chunk_size
                chunks.append(segment[:chunk_size])
                current_chunk = segment[chunk_size-chunk_overlap:]
    
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    # Filtrar chunks vacíos
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
st.title('📜 Generación Aumentada por Recuperación (RAG) - Texto Crudo')
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
    **Modo Sin Dependencias Externas:** Este Agente ahora requiere que **PEGUES** el texto completo del documento aquí abajo, ya que la lectura de archivos PDF necesita librerías externas.
    
    *Utiliza la API de Gemini para la vectorización y la respuesta.*
    """)

# Nuevo input: Área de texto para pegar el contenido del documento
document_text = st.text_area("✍️ Pega el Texto Completo del Documento Aquí", height=300, key="doc_text")

# Procesar el texto si ha sido pegado
if document_text:
    try:
        # Extraer el texto del Papiro Digital (ya lo tenemos en document_text)
        text = document_text
        
        if not text.strip():
             st.error("El Área de Texto no contiene contenido legible.")
             st.stop()
        
        st.info(f"Papiro Descifrado: **{len(text)}** caracteres")
        
        # Generar un ID simple para el texto pegado
        text_hash = hash(text)

        # --- Lógica de Almacenamiento y Recuperación (Vectorización en session_state) ---
        # Si la base de conocimiento no existe o el texto ha cambiado, la creamos
        if 'chunk_embeddings' not in st.session_state or st.session_state.get('text_hash') != text_hash:
            
            # 1. Dividir el texto en fragmentos (Ritos de Fragmentación)
            with st.spinner("Fragmentando el Códice para la Memoria Arcana..."):
                chunks = split_text_into_chunks(text, chunk_size=500, chunk_overlap=20)
            
            st.success(f"Códice Fragmentado en **{len(chunks)}** secciones de Memoria")

            # 2. Crear embeddings (El Altar de la Memoria)
            with st.spinner("Inscribiendo Fragmentos en el Altar de la Memoria (Vectorización)..."):
                
                chunk_embeddings = []
                for i, chunk in enumerate(chunks):
                    # Solo llama al embedding si la aplicación no está en modo de espera
                    if not st.session_state.get('is_rerun_pending', False):
                        st.caption(f"Vectorizando fragmento {i+1}/{len(chunks)}...")
                        
                        # Llamada asíncrona simulada para el embedding
                        embedding = st.async_call(get_gemini_embedding, chunk=chunk)
                        chunk_embeddings.append(embedding)

                # Si hubo llamadas asíncronas pendientes, se actualiza el estado y se detiene la ejecución
                if st.session_state.get('is_rerun_pending', False):
                    st.info("Esperando la finalización de la vectorización. Reintentando...")
                    st.session_state.chunk_texts = chunks
                    st.session_state.text_hash = text_hash
                    st.stop() # Detener la ejecución actual
                    
                # Almacenar chunks y embeddings en la sesión
                st.session_state.chunk_texts = chunks
                st.session_state.chunk_embeddings = np.array(chunk_embeddings)
                st.session_state.text_hash = text_hash
        
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
                query_embedding = st.async_call(get_gemini_embedding, text=user_question)
                
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
                response = st.async_call(get_llm_answer, context=context, question=user_question)
            
            # Mostrar la respuesta
            st.markdown("### 🗣️ Respuesta del Oráculo:")
            st.markdown(response)
                
    except Exception as e:
        # Captura errores de la API (p. ej., clave inválida, límite de rate) o de procesamiento de PDF
        st.error(f"Error Crítico al procesar el Papiro o al llamar a la API: {str(e)}")
        
else:
    st.info("Por favor, pega el texto del documento en el área superior para comenzar la Interrogación.")

# Información adicional y pie de página
st.markdown("---")
st.caption("""
**Notas del Erudito**: Este Códice opera asumiendo que solo se requiere `numpy` además de las librerías estándar de Python y las utilidades de `st.async_call` para la comunicación con la API.
""")
