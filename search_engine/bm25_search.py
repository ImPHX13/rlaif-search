from elasticsearch import Elasticsearch

def bm25_search(es, query, top_k):
    if es is None:
        raise ValueError("Elasticsearch client is not initialized")
    
    try:
        results = es.search(index='documents', body={
            'query': {
                'match': {
                    'content': query
                }
            },
            'size': top_k
        })
        return [hit['_source']['content'] for hit in results['hits']['hits']]
    except Exception as e:
        print(f"Error during search: {e}")
        return []