from pydantic import BaseModel, Field 

from loader import Loader
from vectorstore import VectorDB 
from rag import Rag 


class Input(BaseModel):
    question: str = Field(..., title = "Question")
class Output(BaseModel):
    answer: str = Field(..., title = "Answer") 
    
def build_rag_chain(llm, data_dir, data_type):
    doc_loaded = Loader(file_type = data_type).load_dir(data_dir, workers = 1)
    retriever = VectorDB(documents = doc_loaded).get_retriever()
    rag_chain = Rag(llm).get_chain(retriever) 
    
    return rag_chain 
    
    
