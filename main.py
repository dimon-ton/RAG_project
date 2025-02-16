import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Missing OpenAI API Key. Please check your .env file.")

PDF_FOLDER = "data/pdf_files"
CHROMA_PATH = "vector_db_pdf"
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.7
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def load_pdf_documents(folder_path):
    pdf_docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(folder_path, filename))
            try:
                pdf_docs.extend(loader.load())
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return pdf_docs

pdf_docs = load_pdf_documents(PDF_FOLDER)
if not pdf_docs:
    raise ValueError("No PDF documents found in the folder.")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(pdf_docs)
print(f"Total chunks created: {len(chunks)}")

embedding_model = OpenAIEmbeddings()
vector_db = Chroma.from_documents(
    chunks, embedding_model, persist_directory=CHROMA_PATH
)
print("Vector DB created successfully.")

retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 3})

def query_documents(query):
    retrieved_docs = retriever.invoke(query)
    final_docs = retrieved_docs[:2]
    context = "\n".join([doc.page_content for doc in final_docs])[:10000]
    return context

llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE)
qa_chain = RetrievalQA.from_llm(llm=llm, retriever=retriever)

query = "แนะนำอาหารที่มีราคาต่ำกว่า 100 บาท"
context = query_documents(query)
response = qa_chain.invoke(f"{query}\nข้อมูลที่เกี่ยวข้อง:\n{context}")

print("Response:\n", response)