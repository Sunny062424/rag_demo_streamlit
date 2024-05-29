from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_aws.embeddings import BedrockEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory


from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)

# Claude
import boto3
from langchain_aws import ChatBedrock
#from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import streamlit as st
import time

aws_access_key_id = st.secrets["aws_access_key_id"]
aws_secret_access_key = st.secrets["aws_secret_access_key"]

boto_session = boto3.session.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key)

st.set_page_config(page_title="RAG Demo")
st.title("국가공무원 복무규정")
st.caption("🚀 A streamlit chatbot powered by AWS Bedrock Claude LLM")

#모델-config
bedrock_runtime = boto_session.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

model_kwargs =  { 
    "temperature": 0.0,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}

llm = ChatBedrock(
    client=bedrock_runtime,
    model_id=model_id,
    model_kwargs=model_kwargs,
    streaming=True
)

#Get Retriever
def load_vector_db():
    # load db
    #embeddings_model = OpenAIEmbeddings()
    embeddings_model = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",
                                        client=bedrock_runtime)
    vectorestore = FAISS.load_local('./vecdb/faiss', embeddings_model, allow_dangerous_deserialization=True )
    retriever = vectorestore.as_retriever()
    return retriever

retriever = load_vector_db()


#Contextualize question
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


### Answer question ###
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say just say "복무규정에 없는 내용입니다. 다시 질문해주세요." \
Answer correctly using given context. \
{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

### Statefully manage chat history ###
store = {}
greet_msg = '반갑습니다!🖐🏻  \n  \n저는 복무규정 Q&A봇 Sapie입니다.  \n  \n궁금한 사항에 대해 질문해 주세요!'
st.chat_message("ai").write(greet_msg)

def get_session_history(session_id: str) -> StreamlitChatMessageHistory:
    if session_id not in store:
        store[session_id] = StreamlitChatMessageHistory(key="chat_history")
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
#stream_func
def get_response(chain, prompt, config):
    return (
        val for chunk in chain.stream({"input": prompt}, config)
        for key, val in chunk.items() if key == 'answer'
    )

# session_id = 123
history = get_session_history("123")
for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    config = {"configurable": {"session_id": "any"}}
    #response = conversational_rag_chain.invoke({"input": prompt}, config)["answer"]
        #source_documents = response['source_documents']
        #st.chat_message("ai").write(response)    
    st.chat_message("ai").write_stream(get_response(conversational_rag_chain, prompt, config))
        #with st.expander("참고 문서 확인"):
            #st.markdown(source_documents.metadata['source'], help = source_documents.page_content) 
            #st.chat_message("ai").write_stream(response_generator(response))
