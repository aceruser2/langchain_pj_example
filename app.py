from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langserve import add_routes
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from typing import Literal
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)


model = Ollama(
    base_url="http://0.0.0.0:7869",
    model="phi3",
)


class Answer(BaseModel):
    ans: str = Field(description="you ans and you ans only use chinese answer")
    # mood:  str = Field(description="you mood now and the mood only use english answer")
    mood: Literal[
        "Happy",
        "Joyful",
        "Delighted",
        "Sad",
        "Sorrowful",
        "Melancholic",
        "Angry",
        "Furious",
        "Irritated",
        "Fearful",
        "Scared",
        "Terrified",
        "Anxious",
        "Worried",
        "Distressed",
        "Calm",
        "Serene",
        "Tranquil",
        "Joyous",
        "Jubilant",
        "Ecstatic",
        "Grateful",
        "Thankful",
        "Appreciative",
        "Tired",
        "Exhausted",
        "Weary",
        "Excited",
        "Thrilled",
        "Enthusiastic",
        "Chaotic",
        "Confused",
        "Disorganized",
        "Lonely",
        "Isolated",
        "Alone",
        "Disheartened",
        "Dejected",
        "Despondent",
        "Confident",
        "Self-assured",
        "Bold",
        "Elegant",
        "Graceful",
        "Refined",
        "Ashamed",
        "Embarrassed",
        "Humiliated",
        "Curious",
        "Inquisitive",
        "Interested",
        "Cautious",
        "Wary",
        "Careful",
        "Relaxed",
        "Easygoing",
        "Untroubled",
        "Hopeful",
        "Optimistic",
        "Positive",
    ]


embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
)


loader = DirectoryLoader("./data", glob="*.txt", loader_cls=TextLoader)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

documents = loader.load_and_split(text_splitter)
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()


parser = JsonOutputParser(pydantic_object=Answer)


prompt = PromptTemplate(
    template="Answer the question based on on the following context:{context}, Answer the user question.\n{format_instructions}\n{question}\n",
    input_variables=["context", "question"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
# document_chain = prompt | model | parser

context = RunnableLambda(lambda x: {"context": retriever, "question": x})

chain = context | prompt | model | parser

# chain = prompt | model | parser

add_routes(
    app,
    chain,
    path="/any_question",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
