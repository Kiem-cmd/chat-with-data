from typing import Union, List, Literal
import glob
import multiprocessing
from tqdm import tqdm 
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 


def remove_non_utf8_characters(text):
    return ''.join(char for char in text if ord(char) < 128)

class PDFLoader:
    def __init__(self) -> None:
        self.num_processes = multiprocessing.cpu_count()
    def load_file(self, files):
        docs = PyPDFLoader(files, extract_images = False).load()
        for doc in docs:
            doc.page_content = remove_non_utf8_characters(doc.page_content)
        return docs
    def __call__(self, files: List[str], **kwargs):
        num_processes = min(self.num_processes, kwargs["workers"])
        print(f"nums processes: {num_processes}")
        with multiprocessing.Pool(processes = num_processes) as pool:
            doc_loaded = []
            total_files = len(files)
            with tqdm(total = total_files, desc = "Loading PDFs ...", unit = "file") as pbar:
                for result in pool.imap_unordered(self.load_file, files):
                    doc_loaded.extend(result)
                    pbar.update(1)
        return doc_loaded
        
class TextSplitter:
    def __init__(self,
                 separators: List[str] = ["\n\n", "\n"," ", ""],
                 chunk_size = 500,
                 chunk_overlap = 0) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            separators = separators,
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap)
    def __call__(self,documents):
        return self.splitter.split_documents(documents) 

class Loader:
    def __init__(self,
                file_type: str = Literal["pdf"],
                split_kwargs: dict = {"chunk_size":500,"chunk_overlap":0}) -> None:
        assert file_type in ["pdf"], "file type must be pdf"
        self.file_type = file_type
        if file_type == "pdf":
            self.doc_loader = PDFLoader()
        else:
            raise ValueError("file type must be pdf")
        self.doc_splitter = TextSplitter(**split_kwargs)
    def load(self, files: Union[str, List[str]], workers: int = 1):
        if isinstance(files, str):
            files = [files]
        doc_loaded = self.doc_loader(files, workers = workers)
        doc_split = self.doc_splitter(doc_loaded)
        return doc_split 
    def load_dir(self, dir_path: str, workers: int = 1):
        if self.file_type == 'pdf':
            files = glob.glob(f"{dir_path}/*.pdf")
            assert len(files) > 0, f"No {self.file_type} files found in {dir_path}" 
        else:
            raise ValueError("file_type must be pdf")
        return self.load(files, workers) 
        
    
if __name__ == "__main__":
    dir_path = "data"
    loader = Loader("pdf")
    docs = loader.load_dir(dir)
    print(docs[0])