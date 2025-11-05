from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from flask_cors import CORS

# Charger les variables d'environnement (.env)
load_dotenv()

app = Flask(__name__)
CORS(app)  
# Configuration
DB_FAISS_PATH = "vectorstore/db_faiss"

# Charger la base FAISS une seule fois
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

vectorstore = get_vectorstore()

# Initialiser le modèle Groq
'''Add your API KEY HERE'''

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5,
    max_tokens=512,
    api_key="API_KEY_HERE",
)

# Prompt RAG
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

# Créer la chaîne RAG
combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
rag_chain = create_retrieval_chain(vectorstore.as_retriever(search_kwargs={"k": 3}), combine_docs_chain)


conversation_history = []  # stocke les messages entre le user et le bot

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()
        frontend_history = data.get("conversationHistory", [])
        print("Message reçu :", user_message)

        if not user_message:
            return jsonify({"error": "Aucun message reçu."}), 400

        # Fusionner l'historique du frontend avec celui du serveur
        for msg in frontend_history:
            if msg not in conversation_history:
                conversation_history.append({"role": msg["role"], "message": msg["content"]})

        # Ajouter le message utilisateur actuel
        conversation_history.append({"role": "user", "message": user_message})

        # Construire le contexte des 5 derniers messages
        previous_context = "\n".join(
            [f"{msg['role'].capitalize()}: {msg['message']}" for msg in conversation_history[-5:]]
        )

        # Exécution du RAG
        response = rag_chain.invoke({"input": f"{previous_context}\nUser: {user_message}"})
        answer = response.get("answer", "Désolé, je n'ai pas de réponse disponible.")

        # Ajouter réponse modèle
        conversation_history.append({"role": "assistant", "message": answer})

        return jsonify({
            "response": answer,
            "conversation_history": conversation_history[-10:]
        })

    except Exception as e:
        print("❌ Erreur serveur :", e)
        return jsonify({"error": str(e)}), 500



@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "clsAPI RAG + Groq opérationnelle sur /chat"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True) 
