"""
Document Ingestion Pipeline
Reads from Azure Blob Storage → chunks → embeds → pushes to Azure AI Search

Flow:
Azure Blob Storage (policy-docs container)
    ↓ download each .txt file
chunk_text() — split into 500-word chunks with 50-word overlap
    ↓
get_embedding() — convert each chunk to 3072-number vector
    ↓
Azure AI Search (healthcare-docs index)
    ↓
Agent can now search and retrieve relevant chunks for any question

Run once to build the knowledge base.
Run again whenever new policy documents are added to Blob Storage.
"""

import os
from openai import OpenAI
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticSearch,
    SemanticPrioritizedFields,
    SemanticField
)
from azure.storage.blob import BlobServiceClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv()

# --- OpenAI client ---
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Azure AI Search clients ---
search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_key = os.getenv("AZURE_SEARCH_KEY")
index_name = os.getenv("AZURE_SEARCH_INDEX", "healthcare-docs")
credential = AzureKeyCredential(search_key)

index_client = SearchIndexClient(
    endpoint=search_endpoint,
    credential=credential
)

search_client = SearchClient(
    endpoint=search_endpoint,
    index_name=index_name,
    credential=credential
)

# --- Azure Blob Storage client ---
blob_service = BlobServiceClient.from_connection_string(
    os.getenv("AZURE_STORAGE_CONNECTION_STRING")
)
container_name = os.getenv("AZURE_STORAGE_CONTAINER", "policy-docs")
container_client = blob_service.get_container_client(container_name)


def create_index():
    """
    Creates the Azure AI Search index schema.

    Like CREATE TABLE in SQL.
    Defines what columns exist before inserting any data.

    Fields:
    - id: unique key for each chunk
    - content: the actual text (searchable by keyword)
    - source_file: which blob it came from (filterable)
    - chunk_number: position in document (filterable)
    - content_vector: 3072-dimension vector (searchable by meaning)
    """
    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            analyzer_name="en.microsoft"
        ),
        SimpleField(
            name="source_file",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True
        ),
        SimpleField(
            name="chunk_number",
            type=SearchFieldDataType.Int32,
            filterable=True
        ),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(
                SearchFieldDataType.Single
            ),
            searchable=True,
            vector_search_dimensions=3072,
            vector_search_profile_name="hnsw-profile"
        )
    ]

    # HNSW = fast approximate nearest neighbor algorithm
    # Builds a graph so search compares query to ~200 chunks
    # instead of ALL chunks — makes search instant at scale
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(name="hnsw-algo")
        ],
        profiles=[
            VectorSearchProfile(
                name="hnsw-profile",
                algorithm_configuration_name="hnsw-algo"
            )
        ]
    )

    # Semantic search re-ranks top results using language understanding
    # Goes beyond keyword matching — understands context
    semantic_config = SemanticConfiguration(
        name="semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            content_fields=[SemanticField(field_name="content")]
        )
    )

    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=SemanticSearch(
            configurations=[semantic_config]
        )
    )

    result = index_client.create_or_update_index(index)
    print(f"Index created/updated: {result.name}")
    return result


def chunk_text(text: str, chunk_size: int = 500,
               overlap: int = 50) -> list[str]:
    """
    Split document into overlapping word chunks.

    chunk_size=500: each chunk is ~500 words
    overlap=50: last 50 words of chunk N = first 50 words of chunk N+1

    Why overlap?
    Without overlap: a sentence split across chunk boundary is lost.
    With overlap: every sentence appears in at least one complete chunk.

    Example (simplified with chunk_size=5, overlap=2):
    Text: "A B C D E F G H"
    Chunk 1: "A B C D E"
    Chunk 2: "D E F G H"  ← D and E repeated for context continuity
    """
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def get_embedding(text: str) -> list[float]:
    """
    Convert text to a 3072-dimension vector using OpenAI embeddings.
    text-embedding-3-large produces the most accurate vectors
    for semantic similarity tasks.
    """
    response = openai_client.embeddings.create(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
        input=text
    )
    return response.data[0].embedding


def ingest_documents():
    """
    Main ingestion pipeline.

    Step 1: List all .txt files in Azure Blob Storage policy-docs container
    Step 2: Download each file
    Step 3: Split into 500-word chunks with 50-word overlap
    Step 4: Embed each chunk (3072 numbers per chunk)
    Step 5: Upload all chunks to Azure AI Search in batches of 100

    After this runs:
    - Azure AI Search contains all policy chunks
    - Each chunk has a vector embedding
    - Agent can find relevant chunks for any question in milliseconds
    """
    documents = []
    chunk_id = 0

    # Step 1: List all blobs in container
    blobs = list(container_client.list_blobs())
    txt_blobs = [b for b in blobs if b.name.endswith(".txt")]
    print(f"Found {len(txt_blobs)} documents in Azure Blob Storage")

    for blob in txt_blobs:
        print(f"\nProcessing: {blob.name} ({blob.size} bytes)")

        # Step 2: Download blob content as text
        blob_client = container_client.get_blob_client(blob.name)
        raw_bytes = blob_client.download_blob().readall()
        text = raw_bytes.decode("utf-8")

        # Step 3: Chunk the text
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        print(f"  Chunks created: {len(chunks)}")

        # Step 4: Embed each chunk
        for i, chunk in enumerate(chunks):
            print(f"  Embedding chunk {i+1}/{len(chunks)}...", end="\r")

            embedding = get_embedding(chunk)

            documents.append({
                "id": f"chunk_{chunk_id:04d}",
                "content": chunk,
                "source_file": blob.name,
                "chunk_number": i,
                "content_vector": embedding
            })
            chunk_id += 1

        print(f"  Done: {len(chunks)} chunks embedded        ")

    # Step 5: Upload to Azure AI Search in batches
    print(f"\nUploading {len(documents)} chunks to Azure AI Search...")

    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        result = search_client.upload_documents(documents=batch)
        print(f"Uploaded batch {i//batch_size + 1}: {len(batch)} chunks")

    print(f"\n✅ Ingestion complete!")
    print(f"   Total chunks indexed: {len(documents)}")
    print(f"   Source: Azure Blob Storage → {container_name}")
    print(f"   Destination: Azure AI Search → {index_name}")
    print(f"   Endpoint: {search_endpoint}")


if __name__ == "__main__":
    print("=== HEALTHCARE DOCUMENT INGESTION ===")
    print(f"Source: Azure Blob Storage ({container_name})")
    print(f"Destination: Azure AI Search ({index_name})")
    print()

    print("Step 1: Creating search index...")
    create_index()

    print("\nStep 2: Ingesting documents from Blob Storage...")
    ingest_documents()