import re
import os
import ollama
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Configuración de la barra lateral
with st.sidebar:
    st.title('🎣 PezChat: Tu Asistente de Pesca')
    st.markdown("**Modelo utilizado:** llama3.2:1b")
    st.write("Este chatbot utiliza el modelo llama 3.2:1b para proporcionar respuestas sobre temas de pesca.")

# Iniciar historial de conversación y nombre de usuario
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "nombre" not in st.session_state:
    st.session_state.nombre = None

# Inicia la conversación
if not st.session_state.star_chat:
    st.write('Presiona "Iniciar Chat" para comenzar')
else:
    if not st.session_state.input_text:
        st.write('Por favor ingresa tu consulta')
    else:
        # Captura el nombre del usuario
        if "mi nombre es" in st.session_state.input_text.lower():
            st.session_state.nombre = st.session_state.input_text.split("mi nombre es")[1].strip()
            st.session_state.chat_history.append(f"Bot: ¡Hola, {st.session_state.nombre}!")
        
        # Prompt 
        prompt = f"""
        Basado en esta información sobre pesca: '{st.session_state.input_text}', Tu tarea es proporcionar respuestas claras, detalladas y útiles a cualquier consulta relacionada con la pesca, sin importar el nivel de conocimiento del usuario. Asegúrate de:

        1. Responder de manera concisa, adaptando el nivel de detalle a la complejidad de la consulta.
        2. Incluir ejemplos, datos relevantes o recomendaciones útiles, cuando sea necesario, para enriquecer la respuesta.
        3. Si el usuario solicita información sobre técnicas, equipos, lugares de pesca, especies o cualquier otro aspecto relacionado con la pesca, proporciona información precisa y clara.
        4. Si el usuario pregunta algo relacionado con el código o visualizaciones (como en una consulta técnica), también sé capaz de proporcionar ejemplos o sugerencias.
        5. No des por sentada la experiencia del usuario en pesca. Sé amable y accesible con todos los niveles de conocimiento.

        Cuando respondas, evita información innecesaria que pueda resultar confusa. Prioriza siempre la claridad y la aplicabilidad práctica.
        """

        # Genera la respuesta basada en el modelo seleccionado
        message = "Respuesta no generada aún."
        if st.session_state.model == 'gpt-4o':
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": "Eres un asistente útil especialista y unicamente en temas de pesca."},
                          {"role": "user", "content": prompt}]
            )
            message = completion.choices[0].message.content
        else:
            response = ollama.chat(model=st.session_state.model, messages=[{'role': 'user', 'content': prompt}])
            message = response['message']['content']

        # Muestra el informe y el código en Streamlit
        st.markdown("## Informe y Código Generado")
        st.write(message)

        # Extrae y ejecuta el código Python, si está presente
        pattern = r"```python(.*?)```"
        match = re.search(pattern, message, re.DOTALL)
        
        st.markdown("## Visualización del Código")
        if match:
            code = match.group(1).strip()
            st.markdown(f"```python\n{code}\n```")
            exec(code)  # Solo para demostración, asegúrate de revisar el código antes de ejecutarlo en producción
        else:
            st.write("No se encontró código Python.")
