# app.py
import os
import gradio as gr

# --- Importaciones de LangChain ---
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

print("Iniciando Asistente Experto de SUTEV...")
rag_chain = None
initialization_error = None

try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("ERROR: GOOGLE_API_KEY no está configurada.")

    loader = TextLoader("estatutos.txt", encoding="utf-8")
    docs = loader.load()
    if not docs:
        raise ValueError("El archivo 'estatutos.txt' está vacío.")

    headers_to_split_on = [("#", "Capitulo"), ("##", "Seccion")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    splits = markdown_splitter.split_text(docs[0].page_content)
    if not splits:
        raise ValueError("No se generaron fragmentos del documento.")

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)

    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
    keyword_retriever = BM25Retriever.from_documents(splits)
    keyword_retriever.k = 7

    ensemble_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, keyword_retriever], weights=[0.75, 0.25]
    )

    reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=reranker_model, top_n=4)
    reranking_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )

    # ⚠️ Corrección: gemini-2.5-flash no existe → usa gemini-1.5-flash
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.05, google_api_key=api_key)

    final_prompt_template = """
    Eres "Compañero Asesor", un asistente de IA experto en los estatutos de SUTEV...
    (tu prompt completo aquí, igual que antes)
    **Pregunta del Afiliado:** {input}
    **Respuesta Experta (Sintetizada y Citada):**
    """
    final_prompt = PromptTemplate.from_template(final_prompt_template)

    question_answer_chain = create_stuff_documents_chain(llm, final_prompt)
    rag_chain = create_retrieval_chain(reranking_retriever, question_answer_chain)
    print("✅ Asistente listo!")

except Exception as e:
    initialization_error = str(e)
    print(f"❌ Error: {e}")

def get_response(message, history):
    if initialization_error:
        return f"Error de inicio: {initialization_error}"
    if not rag_chain:
        return "El asistente no está disponible."
    try:
        response = rag_chain.invoke({"input": message})
        return response["answer"]
    except Exception as e:
        return f"Error al procesar: {str(e)}"

# Lanzar Gradio en el puerto de Render
demo = gr.ChatInterface(
    fn=get_response,
    title="Asistente SUTEV - Prueba Web",
    description="Pregúntame sobre los estatutos del sindicato.",
    examples=["¿Cuáles son los deberes de un afiliado?", "¿Cómo se elige la Junta Directiva?"],
    theme="soft"
)

if __name__ == "__main__":
    # Render usa el puerto definido en la variable PORT
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)