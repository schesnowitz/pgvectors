from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
embeddings = OpenAIEmbeddings()
loader = TextLoader('royalcaribbean.txt', encoding='utf-8')
documents = loader.load()

# print(documents) 
# print(len(documents)) 

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# print(texts[0])
# print(len(texts))
embeddings = OpenAIEmbeddings()
# vector = embeddings.embed_query('Testing the embedding model')

# print(len(vector))  # 1536 dimensions

doc_vectors = embeddings.embed_documents([t.page_content for t in texts[:5]])

print(len(doc_vectors))  # 5 vectors in the output
# print(doc_vectors[0])    # this will output the first chunk's 1539-dimensional vector