import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from fuzzywuzzy import fuzz
from gensim import corpora
from gensim.summarization.bm25 import BM25
import sys
sys.path.insert(1, '../')
from repobench.metrics import accuracy_at_k


def run_all(data, snippets_col, query_col, result_file):
    try:
        res = pd.read_csv(result_file)
    except:
        res = pd.DataFrame()
    print(res.columns)
    gold_snippets = [data[i]['gold_snippet_index'] for i in range(len(data))]
    tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
    def write_res(header, preds):
        col = []
        for k in range(1, 6, 2):
            col.append(accuracy_at_k(preds, gold_snippets, k=k))
        res[header] = col
        print(f'{header} is done')
        res.to_csv(result_file, index=False)

    if not set(['Random', 'Jaccard', 'ES', 'BM25']) <= set(res.columns):
        for row in data:
            row[f'{query_col}_tokenized'] = [tokenizer.decode(i) for i in tokenizer.encode(row[query_col])]
            row[f'{snippets_col}_tokenized'] = [[tokenizer.decode(i) for i in tokenizer.encode(c)] for c in row[snippets_col]]
    
    if 'Random' not in res.columns:
        y_pred_rand = []
        for i in range(len(data)):
            y_pred_rand.append(np.random.permutation(range(len(data[i][snippets_col]))))
        write_res('Random', y_pred_rand)

    if 'Jaccard' not in res.columns:
        def jaccard_similarity(doc1, doc2): 
            doc1 = set(doc1)
            doc2 = set(doc2)
            intersection = doc1.intersection(doc2)
            union = doc1.union(doc2)
            return float(len(intersection)) / len(union)
    
        y_pred_jac = []
        for i in range(len(data)):
            dists = []
            ids_next_line = data[i][f'{query_col}_tokenized']
            for num, snippet in enumerate(data[i][f'{snippets_col}_tokenized']):
                dists.append(jaccard_similarity(snippet, ids_next_line))
            y_pred_jac.append(np.argsort(dists)[::-1])
        write_res('Jaccard', y_pred_jac)

    if 'ES' not in res.columns:
        y_pred_es = []
        for i in range(len(data)):
            dists = []
            ids_next_line = data[i][f'{query_col}_tokenized']
            for num, snippet in enumerate(data[i][f'{snippets_col}_tokenized']):
                dists.append(fuzz.ratio(snippet, ids_next_line))
            y_pred_es.append(np.argsort(dists)[::-1])
        write_res('ES', y_pred_es)

    if 'BM25' not in res.columns:
        corpus = []
        for sample in tqdm(data):
            corpus += sample[f'{snippets_col}_tokenized']
        bm25 = BM25(corpus)
    
        ind = 0
        y_pred_bm25 = []
        for sample in data:
            num_snippets = len(sample[f'{snippets_col}_tokenized'])
            dists = bm25.get_scores(sample[f'{query_col}_tokenized'])[ind:ind+num_snippets]
            ind+=num_snippets
            y_pred_bm25.append(np.argsort(dists)[::-1])
        write_res('BM25', y_pred_bm25)

    if 'Unixcoder' not in res.columns:
        from unixcoder import UniXcoder
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UniXcoder("microsoft/unixcoder-base")
        model.to(device)
        def unixcoder_encode(snippet):
            tokens_ids = model.tokenize([snippet],max_length=512,mode="<encoder-only>")
            return torch.tensor(model(torch.tensor(tokens_ids).to(device))[1])
    
        y_pred_unixcoder = []
        for i in range(len(data)):
            # Corpus with example sentences
            corpus_embeddings = torch.cat(list(map(unixcoder_encode, data[i][snippets_col])))
            # Query sentences:
            query_embedding = unixcoder_encode(data[i][query_col])
            cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
            y_pred_unixcoder.append(torch.argsort(cos_scores, descending=True))
        write_res('Unixcoder', y_pred_unixcoder)

    if 'msmarco-MiniLM-L-6-v3' not in res.columns:
        embedder = SentenceTransformer('msmarco-MiniLM-L-6-v3')
        y_pred_all_mini = []
        for i in range(len(data)):
            # Corpus with example sentences
            corpus_embeddings = embedder.encode(data[i][snippets_col], convert_to_tensor=True)
            # Query sentences:
            query_embedding = embedder.encode(data[i][query_col], convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
            y_pred_all_mini.append(torch.argsort(cos_scores, descending=True))
        write_res('msmarco-MiniLM-L-6-v3', y_pred_all_mini)

    if 'msmarco-MiniLM-L-12-v3' not in res.columns:
        embedder = SentenceTransformer('msmarco-MiniLM-L-12-v3')
        y_pred_all_mini = []
        for i in range(len(data)):
            # Corpus with example sentences
            corpus_embeddings = embedder.encode(data[i][snippets_col], convert_to_tensor=True)
            # Query sentences:
            query_embedding = embedder.encode(data[i][query_col], convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
            y_pred_all_mini.append(torch.argsort(cos_scores, descending=True))
        write_res('msmarco-MiniLM-L-12-v3', y_pred_all_mini)

    if 'msmarco-distilbert-base-v3' not in res.columns:
        embedder = SentenceTransformer('msmarco-distilbert-base-v3')
        y_pred_all_mini = []
        for i in range(len(data)):
            # Corpus with example sentences
            corpus_embeddings = embedder.encode(data[i][snippets_col], convert_to_tensor=True)
            # Query sentences:
            query_embedding = embedder.encode(data[i][query_col], convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
            y_pred_all_mini.append(torch.argsort(cos_scores, descending=True))
        write_res('msmarco-distilbert-base-v3', y_pred_all_mini)

    if 'msmarco-distilbert-base-v4' not in res.columns:
        embedder = SentenceTransformer('msmarco-distilbert-base-v4')
        y_pred_all_mini = []
        for i in range(len(data)):
            # Corpus with example sentences
            corpus_embeddings = embedder.encode(data[i][snippets_col], convert_to_tensor=True)
            # Query sentences:
            query_embedding = embedder.encode(data[i][query_col], convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
            y_pred_all_mini.append(torch.argsort(cos_scores, descending=True))
        write_res('msmarco-distilbert-base-v4', y_pred_all_mini)

    if 'msmarco-roberta-base-v3' not in res.columns:
        embedder = SentenceTransformer('msmarco-roberta-base-v3')
        y_pred_all_mini = []
        for i in range(len(data)):
            # Corpus with example sentences
            corpus_embeddings = embedder.encode(data[i][snippets_col], convert_to_tensor=True)
            # Query sentences:
            query_embedding = embedder.encode(data[i][query_col], convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
            y_pred_all_mini.append(torch.argsort(cos_scores, descending=True))
        write_res('msmarco-roberta-base-v3', y_pred_all_mini)

    if 'msmarco-distilbert-base-dot-prod-v3' not in res.columns:
        embedder = SentenceTransformer('msmarco-distilbert-base-dot-prod-v3')
        y_pred_all_mini = []
        for i in range(len(data)):
            # Corpus with example sentences
            corpus_embeddings = embedder.encode(data[i][snippets_col], convert_to_tensor=True)
            # Query sentences:
            query_embedding = embedder.encode(data[i][query_col], convert_to_tensor=True)
            cos_scores = util.dot_score(query_embedding, corpus_embeddings)[0]
            y_pred_all_mini.append(torch.argsort(cos_scores, descending=True))
        write_res('msmarco-distilbert-base-dot-prod-v3', y_pred_all_mini)

    if 'msmarco-roberta-base-ance-firstp' not in res.columns:
        embedder = SentenceTransformer('msmarco-roberta-base-ance-firstp')
        y_pred_all_mini = []
        for i in range(len(data)):
            # Corpus with example sentences
            corpus_embeddings = embedder.encode(data[i][snippets_col], convert_to_tensor=True)
            # Query sentences:
            query_embedding = embedder.encode(data[i][query_col], convert_to_tensor=True)
            cos_scores = util.dot_score(query_embedding, corpus_embeddings)[0]
            y_pred_all_mini.append(torch.argsort(cos_scores, descending=True))
        write_res('msmarco-roberta-base-ance-firstp', y_pred_all_mini)

    if 'msmarco-distilbert-base-tas-b' not in res.columns:
        embedder = SentenceTransformer('msmarco-distilbert-base-tas-b')
        y_pred_all_mini = []
        for i in range(len(data)):
            # Corpus with example sentences
            corpus_embeddings = embedder.encode(data[i][snippets_col], convert_to_tensor=True)
            # Query sentences:
            query_embedding = embedder.encode(data[i][query_col], convert_to_tensor=True)
            cos_scores = util.dot_score(query_embedding, corpus_embeddings)[0]
            y_pred_all_mini.append(torch.argsort(cos_scores, descending=True))
        write_res('msmarco-distilbert-base-tas-b', y_pred_all_mini)

    if 'llmrails/ember-v1' not in res.columns:
        embedder = SentenceTransformer('llmrails/ember-v1')
        y_pred_ember = []
        for i in range(len(data)):
            # Corpus with example sentences
            corpus_embeddings = embedder.encode(['query: ' + snippet for snippet in data[i][snippets_col]], convert_to_tensor=True)
            # Query sentences:
            query_embedding = embedder.encode('query: '+data[i][query_col], convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
            y_pred_ember.append(torch.argsort(cos_scores, descending=True))
        write_res('llmrails/ember-v1', y_pred_ember)

    if 'thenlper/gte-large' not in res.columns:
        embedder = SentenceTransformer('thenlper/gte-large')
        y_pred_gte = []
        for i in range(len(data)):
            # Corpus with example sentences
            corpus_embeddings = embedder.encode(data[i][snippets_col], convert_to_tensor=True)
            # Query sentences:
            query_embedding = embedder.encode(data[i][query_col], convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
            y_pred_gte.append(torch.argsort(cos_scores, descending=True))
        write_res('thenlper/gte-large', y_pred_gte)
    if 'e5-large' not in res.columns:
        embedder = SentenceTransformer('intfloat/e5-large')
        y_pred_e5_large = []
        for i in range(len(data)):
            # Corpus with example sentences
            corpus_embeddings = embedder.encode(['query: ' + snippet for snippet in data[i][snippets_col]], convert_to_tensor=True)
            # Query sentences:
            query_embedding = embedder.encode('query: '+data[i][query_col], convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
            y_pred_e5_large.append(torch.argsort(cos_scores, descending=True))
        write_res('e5-large', y_pred_e5_large)

    if 'e5-large-v2' not in res.columns:
        embedder = SentenceTransformer('intfloat/e5-large-v2')
        y_pred_e5_large_v2 = []
        for i in range(len(data)):
            # Corpus with example sentences
            corpus_embeddings = embedder.encode(['passage: ' + snippet for snippet in data[i][snippets_col]], convert_to_tensor=True)
            # Query sentences:
            query_embedding = embedder.encode('query: '+data[i][query_col], convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
            y_pred_e5_large_v2.append(torch.argsort(cos_scores, descending=True))
        write_res('e5-large-v2', y_pred_e5_large_v2)
