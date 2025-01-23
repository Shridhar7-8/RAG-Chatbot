import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback
import logging
import re
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate,ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],  
)


class MyEmbeddings:
    def __init__(self, model_name="dunzhang/stella_en_1.5B_v5"):
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device="cuda")

    def __call__(self, text):
        if isinstance(text, str):
            return self.model.encode(text)

    def embed_documents(self, documents):
        return self.model.encode(documents)

    def embed_query(self, query):
        return self.model.encode(query)
    

embedding_model = MyEmbeddings()

vectorstore = FAISS.load_local(
    '/mnt/32mins/ngp/admin/vector_store.bin',
    embedding_model,
    allow_dangerous_deserialization=True
)

llm = ChatOllama(
    model="hf.co/keetrap/llama_3.1_q4:latest",
    temperature=0,
)



def refine_query(query):
    """ This agent refines the query of the user """
    
    refined_query_prompt = ChatPromptTemplate.from_template(
        """
        You are a query refinement assistant specialized in National Mission on Interdisciplinary Cyber-Physical 
        Systems (NM-ICPS). Your job is to refine the given query to make it more clear, concise, and appropriate 
        for retrieval or processing.

        Instructions:
        1. Maintain the original intent of the query.
        2. Remove any ambiguity or vagueness in the phrasing.
        3. Make the query more specific, if possible, to improve its clarity and focus.
        4. Ensure that the query is grammatically correct and well-structured.
        5. If possible, reframe the query as a clear, actionable question.

        Original Query: "{query}"

        Refined Query:
        """
    )

    messages = refined_query_prompt.format_messages(query=query)

    response = llm(messages)
    refined_query = response

    return refined_query


def bot_answer(llm, vectorstore, query):
    prompt_template = """
    You are a document assistant that helps users find information in a context.
    Please provide the most accurate response based on the context and inputs.
    Only provide information that is in the context, not in general.

    Context:
    {context}

    Question:
    {question}

    Assistant:
    """

    prompt_template_name = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template_name}
    )

    result = qa({"query": query})
    print(result)
    assistant_answer = result.get("result", "").strip()
    if "Assistant:" in assistant_answer:
        assistant_answer = assistant_answer.split("Assistant:")[1].strip()

    if "Question:" in assistant_answer:
        assistant_answer = assistant_answer.split("Question:")[0].strip()

    source_info = []
    for doc in result.get("source_documents", [])[:3]:
        page = doc.metadata.get("page", "Unknown page")
        document_name = doc.metadata.get("source", "Unknown document")
        source_info.append(f"Page {page+1}, Document: {document_name}")

   
    source_text = "\n".join(source_info)
    full_response = f"{assistant_answer}\n\nSources:\n{source_text}"

    return full_response



class Question(BaseModel):
    query: str



@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI-powered Document Assistant!"}



@app.post("/ask-question/")
async def ask_question(question: Question):
    query = question.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    casual_responses = {
        "hi": "Hi! I am good. How can I help you?",
        "hello": "Hello! How can I assist you?",
        "how are you": "I am just a bot, but I'm doing great! How can I help you?"
    }
    
    for key in casual_responses:
        if key in query.lower():
            return {"response": casual_responses[key]}
        
    try:
        refined_query = refine_query(query)
        refined_query = refined_query.content.strip()
        match = re.search(r'"([^"]+)"', refined_query)
        refined_query = match.group(1) if match else refined_query
        print(refined_query)
        response = bot_answer(llm, vectorstore, refined_query)
        return {"response": response}
    except Exception as e:
        logging.error("Error occurred: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/create-item/")
def create_item(item: dict):
    return {"message": "Item created successfully!", "item": item}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8010, reload=True)
