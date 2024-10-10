import json
from elasticsearch import Elasticsearch
import ssl
from urllib3.exceptions import InsecureRequestWarning
import urllib3

# Suppress only the single warning from urllib3 needed.
urllib3.disable_warnings(InsecureRequestWarning)

def setup_elasticsearch(purge=False):
    try:
        es = Elasticsearch(
            ["https://localhost:9200"],
            basic_auth=("elastic", ""),
            verify_certs=False,
            ssl_show_warn=False
        )
        
        if es.ping():
            print("Connected to Elasticsearch")
            print(es.info())
        else:
            print("Could not connect to Elasticsearch")
            return None
        
        # Purge existing index if requested
        if purge and es.indices.exists(index='documents'):
            es.indices.delete(index='documents')
            print("Existing 'documents' index purged")
        
        # Create index with BM25 similarity if it doesn't exist
        if not es.indices.exists(index='documents'):
            es.indices.create(index='documents', body={
                'settings': {
                    'similarity': {
                        'default': {
                            'type': 'BM25'
                        }
                    }
                },
                'mappings': {
                    'properties': {
                        'content': {'type': 'text'}
                    }
                }
            })
            print("Index 'documents' created successfully")
        else:
            print("Index 'documents' already exists")
        
        # Load and index documents
        with open('data/documents.json', 'r') as f:
            documents = json.load(f)
        
        for i, doc in enumerate(documents):
            es.index(index='documents', id=i, body={'content': doc})
        
        print(f"Indexed {len(documents)} documents")
        return es
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
