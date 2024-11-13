import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Cargar configuración de entorno
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = 'us-east-1'
PINECONE_INDEX_NAME = 'pesca-chatbot'

# Crear una instancia de Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Crear o conectarse al índice de Pinecone
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME, 
        dimension=384, 
        metric='cosine', 
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

# Conectar al índice
index = pc.Index(PINECONE_INDEX_NAME)

# Cargar el modelo de embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

def agregar_dato_a_pinecone(texto, id_texto):
    try:
        embedding = model.encode(texto).tolist()
        index.upsert([(id_texto, embedding)])
    except Exception as e:
        print(f"Error al agregar dato a Pinecone: {e}")

def buscar_en_pinecone(consulta, top_k=3):
    try:
        consulta_embedding = model.encode(consulta).tolist()
        resultado = index.query(vector=consulta_embedding, top_k=top_k, include_values=True)
        ids_resultados = [match['id'] for match in resultado['matches']]
        valores_resultados = [match['values'] for match in resultado['matches']]  # Opcional
        return ids_resultados, valores_resultados
    except Exception as e:
        print(f"Error al buscar en Pinecone: {e}")
        return [], []
