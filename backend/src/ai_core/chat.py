from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from . import config, store

# initialise the llm(brain)
llm = ChatGroq(
    model="llama-3.3-70b-versatile", # flash is fast and free-tire eligible
    api_key=config.GROQ_API_KEY, #type: ignore
    temperature=0, #0 means "be factural, don't hallucinate",
)

# Set up the Retriever (the search engine)
# search_kwargs={"k":3} means "fetch the top 3 most relevant chunks"
retriever = store.get_vector_store().as_retriever(search_kwargs={"k":3}) #type: ignore

# define the prompt template (the instructions)
template = """
You are a helpful AI assistant.
Answer the user's question based ONLY on the context provided below.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)

#creating the chain (the pipeline)
# this uses LCEL(LangChain Expression Language)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def ask_ai(question:str):
    """
    This is the main functio nthe fastapi will call

    Args:
        question (str): _description_
    """
    try:
        response = rag_chain.invoke(question)
        return response
    except Exception as e:
        return f"Error processing request: {str(e)}"