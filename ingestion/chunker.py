import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion.table_reconstructor import reconstruct_tables
from config import CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS

def load_and_chunk(txt_path: str) -> list[Document]:
    """
    Load a text file, reconstruct tables, and chunk it.
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Reconstruct tables first
    raw_text = reconstruct_tables(raw_text)

    # Simple section detection
    section_pattern = re.compile(
        r"^(?:Abschnitt|Teil|Kapitel|Anlage|§\s*\d+|\d+[\.\d]*)\s+[^\n]{3,}",
        re.MULTILINE
    )
    
    section_map = []
    for match in section_pattern.finditer(raw_text):
        section_map.append((match.start(), match.group().strip()))

    def get_section_for_offset(offset: int) -> str:
        current = "Unbekannter Abschnitt"
        for pos, heading in section_map:
            if pos <= offset:
                current = heading
            else:
                break
        return current

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
    )

    raw_chunks = splitter.split_text(raw_text)

    documents = []
    cursor = 0
    for i, chunk_text in enumerate(raw_chunks):
        offset = raw_text.find(chunk_text[:60], cursor)
        cursor = offset + 1 if offset != -1 else cursor
        section = get_section_for_offset(offset)

        documents.append(Document(
            page_content=chunk_text,
            metadata={
                "chunk_index": i,
                "section": section,
                "source": txt_path,
                "char_offset": offset
            }
        ))

    return documents
