

def load_prompt_template(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def format_documents(docs):
    # docs is a list of doc dicts: {doc_id, title, text, ...}
    # Keep it simple and just show doc_id and first 200 chars of text
    formatted = []
    for d in docs:
        snippet = d['text'][:200].replace('\n', ' ')
        formatted.append(f"Doc ID: {d['doc_id']}, Title: {d['title']}, Content: {snippet}...")
    return "\n\n".join(formatted)
