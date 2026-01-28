from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from qdrant_client.http.models import VectorParams, PointStruct, Distance

transformer_model = None

def get_transformer_model():
    global transformer_model
    if transformer_model is None:
        transformer_model = SentenceTransformer("all-mpnet-base-v2")
    return transformer_model

qdrant_client = QdrantClient("http://localhost:6333")

def create_qdrant_collection(name:str) -> bool:
    collections = qdrant_client.get_collections().collections
    if name in collections:
        return True
    else:
        qdrant_client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=get_transformer_model().get_sentence_embedding_dimension(), distance=Distance.COSINE)
        )
        return True

class EmbeddingsAndText(BaseModel):
    text: str
    embedding: list[float]

def get_embeddings_and_text(texts:list[str]):
    model = get_transformer_model()
    embeddings_and_text:list[EmbeddingsAndText] = []
    for text in texts:
        embeddings_and_text.append(EmbeddingsAndText(text=text, embedding=model.encode(text).tolist()))
    return embeddings_and_text

def ingest_vectors_in_qdrant(name:str, embeddings_and_text:list[EmbeddingsAndText]):
    qdrant_client.upload_points(collection_name=name, points=[
        PointStruct(
            id=idx, vector=doc.embedding, payload={"text":doc.text}
        )
        for idx, doc in enumerate(embeddings_and_text)
    ],)


def convert_pdf_to_chunks(file_path:str, chunk_size=350, chunk_overlap=50):
    from docling.document_converter import DocumentConverter
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    converter = DocumentConverter()
    result = converter.convert(file_path)
    text = result.document.export_to_text()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                              separators=["\n\n", "\n", " ", ""], length_function=len)
    split_text = splitter.split_text(text)
    print(f"Total chunks: {len(split_text)}")
    return split_text

