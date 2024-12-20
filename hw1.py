import openai
import os
import streamlit as st
from openai import OpenAI
from langchain_openai import AzureChatOpenAI
from os import environ
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
import pypdf
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.schema import Document
import chromadb
chromadb.api.client.SharedSystemClient.clear_system_cache()



client = AzureChatOpenAI(
    api_key=environ['AZURE_OPENAI_API_KEY'],
    api_version="2023-03-15-preview",
    azure_endpoint=environ['AZURE_OPENAI_ENDPOINT'],
    azure_deployment=environ['AZURE_OPENAI_MODEL_DEPLOYMENT'],
)

st.title("📝 File Q&A with OpenAI")
uploaded_files = st.file_uploader("Upload an article", type=("txt", "md", "pdf"), accept_multiple_files=True)

all_documents =[]

#Input file
question = st.chat_input(
    "Ask something about the article",
    disabled=not uploaded_files,
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Ask something about the article"}]


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# read PDFs using pypdf
def read_pdf(pdf_file):
    reader = pypdf.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()
    print(text)
    #print(type(text))
    return text

# Load all documents
all_documents = []

if question and uploaded_files:
    for file in uploaded_files:
        if file.type == "application/pdf":
            text = read_pdf(file)
            document = Document(page_content=text, metadata={"source": file.name})
            all_documents.append(document)

        elif file.type == "text/plain":
            text = file.read().decode("utf-8")
            document = Document(page_content=text, metadata={"source": file.name})
            all_documents.append(document)

    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=0
    )
    chunks = text_splitter.split_documents(all_documents)


    # Show each chunk (optional)
    #st.write("### Document Chunks")
    #for i, chunk in enumerate(chunks):
    #    st.write(f"**Chunk {i+1}:**")
    #    st.text(chunk.page_content)

    # Index chunks into a vector db (chromaDB)
    vectorstore = Chroma.from_documents(documents=chunks, embedding=AzureOpenAIEmbeddings(model="text-embedding-3-large"))

    #init LLM
    openai.api_key = os.environ['AZURE_OPENAI_API_KEY']
    from langchain_openai import ChatOpenAI
    
    llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    temperature=0.8,
    api_version="2023-06-01-preview",
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

    
    
    #Prepare prompt(Augmentation Step)
    from langchain_core.prompts import PromptTemplate

    template = """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        
        Question: {question} 
        
        Context: {context} 
        
        Answer:
    """
    prompt = PromptTemplate.from_template(template)
    from langchain import hub
    prompt = hub.pull("rlm/rag-prompt")

    #retrieval
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    #build RAG chain
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)
    
    response = rag_chain.invoke(question)

    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)




