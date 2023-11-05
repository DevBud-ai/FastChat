from typing import List
from haystack.nodes.search_engine import WebSearch
from haystack.schema import Document

from fastchat.rag.extracting import Extractor
from fastchat.rag.fetching import Fetcher
from fastchat.rag.contriever import ReferenceFilter

retriever_ckpt_path = '/root/WebGLM/download/retriever-pretrained-checkpoint'
filter_max_batch_size = 4

fetcher = Fetcher()
extractor = Extractor()
filter = ReferenceFilter(retriever_ckpt_path, 'cuda', filter_max_batch_size)

def search_query(query):
    # This search uses the default SerperDev provider, so we just need the API key
    ws = WebSearch(api_key="df287bcf40c3687945654bd02de095ff879c4f7a")
    # documents = ws.run(query="what is the current weather in Trivandrum?")
    
    documents: List[Document] = ws.run(query=query)
    print(documents)
    references = []
    for item in documents[0]['documents']:
        print(item.meta)
        print(item.meta['link'])
        references.append({
            "text": item.content,
            "url": item.meta['link']
        })
    
    print(references)

    urls = [result.meta['link'] for result in documents[0]['documents']]
    # titles = {result.url: result.title for result in search_results}
    print("[System] Count of available urls: ", len(urls))
    if len(urls) == 0:
        print("[System] No available urls. Please check your network connection.")
        return None
        
    print("[System] Fetching ...")
    fetch_results = extractor.fetch(urls)
    cnt = sum([len(fetch_results[key]) for key in fetch_results])
    print("[System] Count of available fetch results: ", cnt)
    if cnt == 0:
        print("[System] No available fetch results. Please check playwright or your network.")
        return None
        
    print("[System] Extracting ...")
    data_list = []
    for url in fetch_results:
        extract_results = extractor.extract_by_html2text(fetch_results[url])
        data_list.append({
            "url": url,
            # "title": titles[url],
            "text": "\n".join(extract_results)
        })
        # for value in extract_results:
        #     data_list.append({
        #         "url": url,
        #         # "title": titles[url],
        #         "text": value
        #     })
    print("[System] Count of paragraphs: ", len(data_list))
    if len(data_list) == 0:
        print("[System] No available paragraphs. The references provide no useful information.")
        return None
    
    print("[System] Filtering ...")

    filtered = filter.produce_references(query, data_list, 1)

    if len(filtered):
        return filtered[0]
    
    return []