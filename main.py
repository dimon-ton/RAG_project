import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.schema import Document

# โหลดตัวแปรสภาพแวดล้อม
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Missing OpenAI API Key. Please check your .env file.")

# กำหนดโฟลเดอร์ที่เก็บไฟล์ PDF
PDF_FOLDER = "data/pdf_files"
CHROMA_PATH = "vector_db_pdf"


def load_pdf_documents(folder_path):
    """โหลดเอกสาร PDF จากโฟลเดอร์ที่กำหนด โดยใช้ PyMuPDFLoader เพื่อให้รองรับการอ่านไฟล์ PDF ที่ซับซ้อน"""
    pdf_docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(folder_path, filename))
            try:
                pdf_docs.extend(loader.load())
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return pdf_docs


# โหลดและแบ่งข้อมูล
pdf_docs = load_pdf_documents(PDF_FOLDER)
if not pdf_docs:
    raise ValueError("No PDF documents found in the folder.")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]) 
chunks = splitter.split_documents(pdf_docs)

print(f"Total chunks created: {len(chunks)}")

# สร้าง Vector Database
embedding_model = OpenAIEmbeddings()
vector_db = Chroma.from_documents(chunks, embedding_model, persist_directory=CHROMA_PATH)
print("Vector DB created successfully.")

# โหลด Vector Database เพื่อใช้งาน
vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)
retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 3})

# ตรวจสอบการดึงข้อมูล
query_test = "ราคาพัดกระเพราหมูกี่บาท"
retrieved_docs = retriever.invoke(query_test)
print("Retrieved Documents:")
for i, doc in enumerate(retrieved_docs):
    print(f"{i+1}: {doc.page_content[:500]}\n")

# คัดกรองเฉพาะเอกสารที่เกี่ยวข้อง
DATE_KEYWORDS = ["สอนวันที่", "วันที่", "สอนวัน"]
filtered_docs = [doc for doc in retrieved_docs if any(keyword in doc.page_content for keyword in DATE_KEYWORDS)]

if not filtered_docs:
    print("No relevant documents with dates found.")
    final_docs = retrieved_docs[:2]  # จำกัดจำนวนข้อมูลที่ส่งไปยังโมเดล
else:
    print("Filtered Documents with Date Information:")
    for i, doc in enumerate(filtered_docs):
        print(f"{i+1}: {doc.page_content[:500]}\n")
    final_docs = filtered_docs[:2]  # จำกัดข้อมูลเพื่อไม่ให้เกิด token limit

# ตั้งค่าโมเดล LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
qa_chain = RetrievalQA.from_llm(llm=llm, retriever=retriever)

# ส่งเอกสารที่เลือกไปให้โมเดลโดยตรง
query = "ราคาพัดกระเพราหมูกี่บาท"
context = "\n".join([doc.page_content for doc in final_docs])[:10000]  # จำกัดจำนวน token ที่ส่งไปยัง LLM
response = qa_chain.invoke(f"{query}\nข้อมูลที่เกี่ยวข้อง:\n{context}")

print("Response:\n", response)
