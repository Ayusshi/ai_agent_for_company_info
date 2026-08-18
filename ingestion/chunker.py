from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(pages):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = []

    chunk_id = 0

    for page in pages:

        page_chunks = splitter.split_text(page["text"])

        for chunk in page_chunks:

            chunks.append({
                "text": chunk,
                "page": page["page"],
                "source": page["source"],
                "chunk_id": chunk_id
            })

            chunk_id += 1

    return chunks