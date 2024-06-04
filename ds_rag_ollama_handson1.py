import ollama
import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
bs_kwargs = dict(parse_only=bs4.SoupStrainer(
    class_ = ("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="llama3")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\nContext: {context}"
    response = ollama.chat(model='llama3', messages=[{
    "role":"user", "content": formatted_prompt}])
    return response["message"]["content"]

retriever = vectorstore.as_retriever()

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(question):
    retrived_docs = retriever.invoke(question)
    formatted_context = combine_docs(retrived_docs)
    return ollama_llm(question, formatted_context)

result = rag_chain("What is Task Decomposition")
print(result)
