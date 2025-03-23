import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os
import json
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from langchain.tools import Tool
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()
analyzer = SentimentIntensityAnalyzer()

# Asegúrate de tener tu API key de OpenAI configurada
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="gemma2-9b-it",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

def obtener_sentimiento(texto):
    return analyzer.polarity_scores(texto)

sentiment_tool = Tool(
    name="Sentiment Analysis",
    func=obtener_sentimiento,
    description="Analiza el sentimiento de un texto y devuelve un puntaje de sentimiento."
)

def evaluar_respuesta(texto):
    prompt = f"""
    Evalúa la calidad de la siguiente respuesta de un usuario a una noticia. 
    La respuesta debe ser clara, bien desarrollada y relevante. 
    Responde solo con "completa" o "incompleta".
    
    Respuesta: {texto}
    """
    respuesta = llm.invoke([HumanMessage(content=prompt)])
    resultado = respuesta.content.strip().lower() if respuesta.content else "incompleta"


def asignar_puntuacion(compound, categoria):
    if categoria in ["Ambiental", "Gobernanza", "Riesgo"]:
        return 100 if compound <= -0.03 else 90 if compound <= -0.025 else 80 if compound <= -0.02 else 70 if compound <= -0.015 else 60 if compound <= -0.01 else 50 if compound <= 0 else 40 if compound <= 0.1 else 30 if compound <= 0.2 else 20 if compound <= 0.4 else 10 if compound <= 0.5 else 0
    elif categoria == "Social":
        return 100 if compound >= 0.025 else 90 if compound >= 0.02 else 80 if compound >= 0.015 else 70 if compound >= 0.01 else 60 if compound >= 0.05 else 50 if compound >= 0 else 40 if compound >= -0.01 else 30 if compound >= -0.02 else 20 if compound >= -0.03 else 10 if compound >= -0.04 else 0

noticias = [
    "Repsol, entre las 50 empresas que más responsabilidad histórica tienen en el calentamiento global",
    "Amancio Ortega crea un fondo de 100 millones de euros para los afectados de la dana",
    "Freshly Cosmetics despide a 52 empleados en Reus, el 18% de la plantilla",
    "Wall Street y los mercados globales caen ante la incertidumbre por la guerra comercial y el temor a una recesión",
    "El mercado de criptomonedas se desploma: Bitcoin cae a 80.000 dólares, las altcoins se hunden en medio de una frenética liquidación",
    "Granada retrasa seis meses el inicio de la Zona de Bajas Emisiones, previsto hasta ahora para abril",
    "McDonald's donará a la Fundación Ronald McDonald todas las ganancias por ventas del Big Mac del 6 de diciembre",
    "El Gobierno autoriza a altos cargos públicos a irse a Indra, Escribano, CEOE, Barceló, Iberdrola o Airbus",
    "Las aportaciones a los planes de pensiones caen 10.000 millones en los últimos cuatro años",
]

if "contador" not in st.session_state:
    st.session_state.contador = 0
    st.session_state.reacciones = {}
    st.session_state.historial = []

title = "Chatbot de Análisis de Sentimiento"
st.title(title)

if st.session_state.contador < len(noticias):
    noticia = noticias[st.session_state.contador]
    st.chat_message("assistant").write(f"**Noticia:** {noticia}")
    
    for mensaje in st.session_state.historial:
        st.chat_message(mensaje["role"]).write(mensaje["content"])
    
    user_input = st.chat_input("Escribe tu reacción")
    
    if user_input:
        if not evaluar_respuesta(user_input):
            st.warning("Tu respuesta parece incompleta. Intenta expandir tu opinión.")
        else:
            sentimiento = sentiment_tool.run(user_input)
            compound = sentimiento['compound']
            
            st.session_state.reacciones[st.session_state.contador] = {"texto": user_input, "sentimiento": sentimiento}
            
            st.session_state.historial.append({"role": "user", "content": user_input})
            st.chat_message("user").write(user_input)
            
            analisis_texto = f"**Análisis de Sentimiento:** {sentimiento}"
            st.session_state.historial.append({"role": "assistant", "content": analisis_texto})
            st.chat_message("assistant").write(analisis_texto)
            
            st.session_state.contador += 1
            st.rerun()
else:
    puntuaciones = {"Ambiental": 50, "Social": 50, "Gobernanza": 50, "Riesgo": 50}
    e_scores, s_scores, g_scores, r_scores = [], [], [], []
    
    for i, sentimiento in st.session_state.reacciones.items():
        compound = sentimiento['sentimiento']['compound']
        if i in [0, 5]:
            e_scores.append(asignar_puntuacion(compound, "Ambiental"))
        elif i in [1, 6]:
            s_scores.append(asignar_puntuacion(compound, "Social"))
        elif i in [2, 7]:
            g_scores.append(asignar_puntuacion(compound, "Gobernanza"))
        elif i in [3, 4, 8]:
            r_scores.append(asignar_puntuacion(compound, "Riesgo"))
    
    if e_scores:
        puntuaciones["Ambiental"] = round(sum(e_scores) / len(e_scores))
    if s_scores:
        puntuaciones["Social"] = round(sum(s_scores) / len(s_scores))
    if g_scores:
        puntuaciones["Gobernanza"] = round(sum(g_scores) / len(g_scores))
    if r_scores:
        puntuaciones["Riesgo"] = round(sum(r_scores) / len(r_scores))
    
    st.chat_message("assistant").write("**Perfil del Inversor:**")
    for categoria, puntaje in puntuaciones.items():
        st.chat_message("assistant").write(f"{categoria}: {puntaje}")
    
    fig, ax = plt.subplots()
    ax.bar(puntuaciones.keys(), puntuaciones.values(), color=['green', 'blue', 'purple', 'red'])
    ax.set_ylabel("Puntuación (0-100)")
    ax.set_title("Perfil del Inversor")
    st.pyplot(fig)
