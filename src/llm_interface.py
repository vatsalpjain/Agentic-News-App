from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

#USE
#answer=rag_simple("Where does Mr. Bingley first take up residence?",rag_retriever,llm)

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key,model_name="gemma2-9b-it",temperature=0.1,max_tokens=1024)

#Simple RAG function: retrieve context + generate response
def rag_simple(query,retriever,llm,top_k=3):
    ## retriever the context
    results=retriever.retrieve(query,top_k=top_k)
    context="\n\n".join([doc['content'] for doc in results]) if results else ""
    if not context:
        return "No relevant context found to answer the question."

    ## generate the answwer using GROQ LLM
    prompt="""You have access only to the given extracted text {context} from a storybook.

    Your task:
    1. Provide the answer to the query in consise way {query}.
    Format:
    Answer:
    <your concise answer here>
    """

    response=llm.invoke([prompt.format(context=context,query=query)])
    return response.content

