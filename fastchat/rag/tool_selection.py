from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

from fastchat.rag.serp_search import search_query


embeddings = HuggingFaceEmbeddings()

def create_index():
    current_affairs_prompts = [
        "Provide the latest updates on the ongoing international climate change negotiations.",
        "What are the key economic implications of the recent government budget announcement in my country?",
        "Give me a summary of major global news headlines from the past 24 hours.",
        "Explain the impact of the recent political election results in [Country Name] on foreign policy.",
        "Tell me about any significant developments in the field of technology from the past week."
    ]
    llm_prompts = [
        "Generate a creative short story involving time travel and its consequences.",
        "Summarize the key concepts of Albert Einstein's theory of relativity.",
        "Translate the following English text into French: 'The quick brown fox jumps over the lazy dog.'",
        "What are the potential applications of artificial intelligence in healthcare?",
        "Generate a poem about the beauty of nature and the changing seasons."
    ]

    document_db_prompts = [
        "Retrieve all documents related to customer feedback from the last quarter.",
        "Find historical sales data for product X from the past five years.",
        "Retrieve legal documents related to contract agreements with Facebook.",
        "Search for research papers on the topic of renewable energy sources.",
        "Retrieve user manuals and technical documentation for Google."
    ]


    list_of_documents = []

    for item in current_affairs_prompts:
        list_of_documents.append(Document(page_content=item, metadata={"tool": "search"}))

    for item in llm_prompts:
        list_of_documents.append(Document(page_content=item, metadata={"tool": "llm"}))

    db = FAISS.from_documents(list_of_documents, embeddings)
    db.save_local("faiss_index")


def get_tool(prompt):
    db = FAISS.load_local("faiss_index", embeddings)
    result = db.similarity_search(prompt)
    print(result)
    if len(result):
        return result[0].metadata['tool']
    
    return 'llm'


def get_context(prompt):
    tool = get_tool(prompt)
    
    context = ''

    if tool == 'search':
        result = search_query(prompt)
        context = result['text']
    print(context)
    return context

# create_index()