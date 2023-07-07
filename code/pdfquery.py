import os
from collections import defaultdict
import configparser

import boto3
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from tenacity import retry, wait_random_exponential, stop_after_attempt

from prompts import QuestionAnswerTemplate

# config = configparser.ConfigPapiprser()
# config.read('../config.ini')

bedrock = boto3.client(
    service_name='bedrock',
    region_name='us-east-1',
    endpoint_url='https://bedrock.us-east-1.amazonaws.com'
)

def get_source(docs):
    sources_dict = defaultdict(list)
    for doc in docs:
        source = doc.metadata['source']
        # Only add the page if it's in the metadata
        if 'page' in doc.metadata:
            page = doc.metadata['page'] + 1
            # Group pages by their sources
            sources_dict[source].append(page)
        else:
            sources_dict[source].append(None)
    
    pointers = ''
    for source, pages in sources_dict.items():
        # Format source and corresponding pages
        source_name = os.path.basename(source)  # get file name from the path
        # Remove None values and convert remaining pages to strings
        pages = [str(page) for page in pages if page is not None]
        # If pages exist, include them in the string
        if pages:
            pointers += f'- {source_name} - page(s): {", ".join(pages)}\n'
        else:
            pointers += f'- {source_name}'

    return pointers.strip()  # remove the trailing new line


def concat_sources(docs):
    texts = [x.page_content for x in docs]
    sources = ''
    for indx, text in enumerate(texts):
        sources += f'Source {indx+1}: {text}\n'
    return sources

class PDFQuery:
    def __init__(self) -> None:
        """init sets up Langchain BedRock embeddings and Bedrock LLM
        """
        self.embeddings = BedrockEmbeddings(client=bedrock)

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.llm = Bedrock(
            client=bedrock, 
            model_id="amazon.titan-tg1-large"
            )
        
        self.chain = None
        self.db = None

        self.splitted_documents = []

    @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
    def ask(self, question: str) -> str:
        """accepts user question, queries relevant documents, then passes it to Question-Answer Template.

        Args:
            question (str): _description_

        Returns:
            str: _description_
        """
        if self.chain is None:
            response = "Please, add a document."
        else:
            docs = self.db.get_relevant_documents(question)
            pointers = get_source(docs)
            sources = concat_sources(docs)

            input_data = {'query': question, 'sources': sources}
            response = self.chain.process(input_data).replace('`', '')
            # response = self.chain.run(input_documents=docs, question=question)
            response += f'\n\nInformation sourced from passages: \n{pointers}'
        return response
    

    def ingest(self, file_path: os.PathLike) -> None:
        """load document, chunk, and insert into Chroma.
        
        this is called when a file is uploaded. based on whether the file is a PDF or docx it will call one of Langchains native loaders. The Langchain loaders are easy but dirty. PyPDFLoader automatically includes page number.

        Args:
            file_path (os.PathLike): path to file
        """
        # load for Q-A embeddings
        if file_path.split('.')[-1] == 'pdf':
            loader = PyPDFLoader(file_path)
        elif file_path.split('.')[-1] == 'docx':
            loader = Docx2txtLoader(file_path)
        documents = loader.load()
        splitted_documents = self.text_splitter.split_documents(documents)
        self.splitted_documents += splitted_documents
        self.db = Chroma.from_documents(self.splitted_documents, self.embeddings).as_retriever()
        self.chain = QuestionAnswerTemplate(self.llm)

    def forget(self) -> None:
        self.db = None
        self.chain = None