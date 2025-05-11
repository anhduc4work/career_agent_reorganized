from tqdm import tqdm
import pandas as pd

# 2. Read CSV with pandas
df = pd.read_csv("scholar_data.csv")
print("done read data")


# 4. Connect to PGVector
from langchain_postgres import PGVector
from langchain_ollama import OllamaEmbeddings

connection = "postgresql+psycopg://postgres:postgres@localhost:5432/postgres"
collection_name = "scholar2"

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

# 5. Prepare documents
from langchain_core.documents import Document

docs = []
ids = []

for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing documents"):
    doc = Document(
        page_content=row['TextContent'],
        metadata={
            "workingtime": row.get("WorkingTime", ""),
            "link": row.get("OriginalLink", ""),
            "position": row.get("PositionList", "")
        }
    )
    docs.append(doc)
    ids.append(str(index))  # Ensure ID is a string

# 6. Add documents to vector store
vector_store.add_documents(docs, ids=ids)