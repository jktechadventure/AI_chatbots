from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chroma import Chroma

# Load the webpage
loader = WebBaseLoader("https://newgenmigration.com/visa-types")
docs = loader.load()

# Process and store embeddings
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))

# Query the chatbot
query = "What is a Dependent Child Resident Visa?"
retriever = vectorstore.as_retriever()
response = retriever.get_relevant_documents(query)
print(response)
