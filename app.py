import streamlit as st
import requests

# Configuração da Página
st.set_page_config(page_title="Predicta RAG", page_icon="🏭")
st.title("🏭 Predicta - Assistente de Manutenção Industrial")
st.caption("Faça perguntas sobre telemetria e laudos técnicos dos motores.")

# Inicializa o histórico do chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe as mensagens anteriores
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Campo de digitação
if prompt := st.chat_input("Digite sua pergunta técnica aqui..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Chama o Back-end (FastAPI)
    with st.spinner("Analisando documentos vetoriais..."):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/api/v1/chat",
                json={
                    "message": prompt,
                    "history": st.session_state.messages[:-1]
                }
            )
            response.raise_for_status()

            resposta_ia = response.json().get("reply", "Desculpe, não consegui processar a resposta.")

            st.session_state.messages.append({"role": "assistant", "content": resposta_ia})
            with st.chat_message("assistant"):
                st.markdown(resposta_ia)

        except requests.exceptions.ConnectionError:
            st.error("⚠️ Não foi possível conectar ao Back-end. Verifique se o FastAPI está rodando na porta 8000.")
        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")
