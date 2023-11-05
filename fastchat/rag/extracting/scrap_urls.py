from typing import List, Dict
import requests
import asyncio
import concurrent.futures

def _pre_handle_urls(urls: List[str]) -> List[str]:
    urls_new = []
    for url in urls:
        if url in urls_new or "http://%s"%url in urls_new or "https://%s"%url in urls_new:
            continue
        if not url.startswith("http"):
            url = "http://%s" % url
        urls_new.append(url)
    return urls_new

def fetch_url(url):
    print(url)
    try:
        response = requests.get(url)
        return url, response.text
    except:
        return url, ''

def fetch_all_urls(urls):
    urls = _pre_handle_urls(urls)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_url, urls))

    ret = dict()
    for url, content in results:
        ret[url] = content

    return ret

