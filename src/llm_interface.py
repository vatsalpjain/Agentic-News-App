from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set")

try:
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="gemma2-9b-it",
        temperature=0.1,
        max_tokens=1024
    )
except Exception as e:
    raise RuntimeError(f"Failed to initialize LLM: {e}")

def rag_simple(query, retriever, llm, top_k=3):
    """Generate answer using RAG"""
    try:
        # Get relevant articles
        results = retriever.retrieve(query, top_k=top_k)
        
        if not results:
            return "No relevant articles found in the database."
            
        # Prepare context from retrieved documents
        context = "\n\n".join([doc['content'] for doc in results])
        
        # Generate response using LLM
        prompt = f"""Based on the following context, answer the query concisely and accurately.

Context:
{context}

Query: {query}

Answer:"""

        response = llm.invoke(prompt)
        return response.content
        
    except Exception as e:
        logger.error(f"RAG error: {e}")
        return f"Error generating response: {str(e)}"

