import os
from datasets import load_dataset

def load_pubmed(csv_path):
    pubmed_ds = load_dataset("csv", data_files=csv_path)
    return pubmed_ds

def process_pubmed_row(examples):
    processed_chunks = []
    for i in range(len(examples['abstract_text'])):
        text_content = (
            f"In PubMed abstract {examples['abstract_id'][i]}, "
            f"the {examples['target'][i]} section states: {examples['abstract_text'][i]}"
        )
        metadata = {
            "source": "PubMed",
            "abstract_id": examples['abstract_id'][i],
            "target": examples['target'][i],
            "line_number": examples['line_number'][i]
        }
        processed_chunks.append({
            "content": text_content,
            "metadata": metadata
        })
    return {"processed_data": processed_chunks}

def build_pubmed_chunks(csv_path):
    pubmed_ds = load_pubmed(csv_path)
    pubmed_processed = pubmed_ds.map(
        process_pubmed_row,
        batched=True,
        remove_columns=pubmed_ds['train'].column_names
    )
    pubmed_chunks = list(pubmed_processed['train']['processed_data'])
    return pubmed_chunks


def txt_chunking(text, chunk_size=600, overlap=100, metadata=None):
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""

    for p in paragraphs:
        if len(current_chunk) + len(p) <= chunk_size:
            current_chunk += p + "\n\n"
        else:
            if current_chunk:
                chunks.append({
                    "content": current_chunk.strip(),
                    "metadata": metadata.copy() if metadata else {}
                })
            current_chunk = current_chunk[-overlap:] + p + "\n\n"

    if current_chunk:
        chunks.append({"content": current_chunk.strip(), "metadata": metadata})

    return chunks

def load_txt_folder(folder_path, source_name):
    all_chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            meta = {
                "source": source_name,
                "file_name": filename
            }
            file_chunks = txt_chunking(content, metadata=meta)
            all_chunks.extend(file_chunks)
    return all_chunks


def build_pubmed_map(final_chunks):
    pubmed_map = {}
    for chunk in final_chunks:
        if chunk['metadata'].get('source') == 'PubMed':
            abs_id = chunk['metadata']['abstract_id']
            pubmed_map.setdefault(abs_id, []).append(chunk)
    return pubmed_map
