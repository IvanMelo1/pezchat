import ollama
import streamlit as st
from config import Config
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

# Configuración de Pinecone
PINECONE_API_KEY = 'pcsk_3tEEED_ACAaqn1zMQHXmgjMh8fawHZAdEAwmg54CZ3bgu8UGFyjNwaZv6FjN9uSw86kxz9'
PINECONE_ENVIRONMENT = 'us-east-1'
PINECONE_INDEX_NAME = 'chatbot-pesca'

# Crear instancia de Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Conectarse al índice de Pinecone
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME, 
        dimension=384, 
        metric='cosine', 
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
index = pc.Index(PINECONE_INDEX_NAME)

# Cargar el modelo de embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

def agregar_dato_a_pinecone(texto, id_texto):
    embedding = model.encode(texto).tolist()
    index.upsert([(id_texto, embedding)])

def buscar_en_pinecone(consulta, top_k=3):
    consulta_embedding = model.encode(consulta).tolist()
    resultado = index.query(vector=consulta_embedding, top_k=top_k, include_values=False)
    ids_resultados = [match['id'] for match in resultado['matches']]
    return ids_resultados

def main():
    with st.sidebar:
        st.title('🎣 PezChat: Tu Asistente de Pesca')
        st.markdown("**Modelo utilizado:**")
        st.session_state.model = st.selectbox('Select Model', Config.OLLAMA_MODELS)

    st.title('🎣 PezChat: Tu Asistente de Pesca')
    
    # Si no existe la sesión de mensajes, la inicializamos.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("¿Qué te gustaría saber sobre la pesca?"):
        # Añadimos el prompt del usuario a los mensajes de la sesión.
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Mostramos el mensaje del usuario en la interfaz.
        with st.chat_message("user"):
            st.markdown(prompt)

        # Buscar en Pinecone si la consulta es relevante
        ids_resultados = buscar_en_pinecone(prompt)
        
        # Si hay resultados, los mostramos
        if ids_resultados:
            respuestas_relevantes = [f"Resultado {i+1}: {id_resultado}" for i, id_resultado in enumerate(ids_resultados)]
            respuesta_pinecone = "\n".join(respuestas_relevantes)
        else:
            respuesta_pinecone = "No encontré información relevante en la base de datos de pesca."

        # Generar respuesta con Ollama
        with st.chat_message("assistant"):
            # Concatenar la respuesta de Pinecone con el sistema de respuesta de Ollama
            system_prompt = Config.SYSTEM_PROMPT
            # Aquí utilizamos preprocessed_prompt para un flujo más claro
            preprocessed_prompt = f"{system_prompt}\n\nUsuario: {prompt}\nAsistente: {respuesta_pinecone}"
            
            # Enviar la consulta a Ollama
            response = ollama.chat(
                model=st.session_state.model,
                messages=[{"role": "system", "content": preprocessed_prompt}] + 
                         [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages] 
            )
            
            message_ollama = response['message']['content']
            st.write(message_ollama)

        # Añadir la pregunta y la respuesta del usuario a Pinecone para persistencia en memoria
        agregar_dato_a_pinecone(prompt, id_texto="pregunta_" + str(len(st.session_state.messages)))
        agregar_dato_a_pinecone(message_ollama, id_texto="respuesta_" + str(len(st.session_state.messages)))

        # Añadir la respuesta del asistente a la sesión de mensajes.
        st.session_state.messages.append({"role": "assistant", "content": message_ollama})

if __name__ == '__main__':
    main()
