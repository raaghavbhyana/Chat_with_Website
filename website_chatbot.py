import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import AIMessage,HumanMessage 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


def url_extractor(url):

    loader=WebBaseLoader(web_path=url)
    document=loader.load()
    text_splitter=RecursiveCharacterTextSplitter()
    document_chunks =text_splitter.split_documents(document)
    embeddings=HuggingFaceInferenceAPIEmbeddings(api_key="",model_name="sentence-transformers/all-MiniLM-l6-v2")
    vector_store = FAISS.from_documents(documents=document_chunks,embedding=embeddings)
    return vector_store
   

def get_retriever_chain(vector_store):
    llm=ChatGoogleGenerativeAI(model="gemini-1.5-pro",google_api_key="")
    retriever=vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain

def get_conversational_rag_chain(retriever_chain): 
    
    llm =ChatGoogleGenerativeAI(model="gemini-1.5-flash",google_api_key="")
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_response(user_input):
    retriever_chain = get_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']



st.set_page_config(page_title="Chat with Website",page_icon="")
st.title("Chat With Website")
with st.sidebar:
    st.header("Setting")
    website_url=st.text_input("Enter URL of website")

if website_url is None or website_url=="":
    st.info("Please enter the URL of the website")
else:
           # session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content="Hello, I am a bot. How can I help you?")]
            
        if "vector_store" not in st.session_state:
            st.session_state.vector_store = url_extractor(website_url)    

        # user input
        user_query = st.chat_input("Type your message here...")
        if user_query is not None and user_query != "":
            response = get_response(user_query)
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))
            
        

        # conversation
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)