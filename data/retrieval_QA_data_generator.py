import argparse
import random

from elasticsearch import Elasticsearch
import json
import tqdm

def build_index(index_name, file_path):
    # Connect to the default Elasticsearch instance at localhost:9200
    es = Elasticsearch(hosts="http://localhost:9200")

    # Define the mapping for the index
    mapping = {
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "text": {"type": "text"}
            }
        }
    }

    # Create the index with the specified mapping
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=mapping)

    # index
    with open(file_path, 'r') as file:
        for line in file:
            # please note that the var to body cannot include field "_id"
            doc = json.loads(line)
            example = {'title': doc['title'], 'text': doc['text']}
            res = es.index(index=index_name, body=example)
            print(f"Document indexed with ID: {res['_id']}")

def search_documents(query, index_name, size=10):
    es = Elasticsearch(hosts="http://localhost:9200")
    search_body = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["title", "text"]
            }
        },
        "size": size
    }
    results = es.search(index=index_name, body=search_body)
    return results['hits']['hits']

def generate_2wiki_retrieve_data(sample_number=100, retrieve_number=100, split='test'):
    '''
    data example: {"_id": "8813f87c0bdd11eba7f7acde48001122", "text": "Who is the mother of the director of film Polish-Russian War (Film)?", "metadata": {"answer": "Ma\u0142gorzata Braunek", "answer_id": "Q274277", "ctxs": [["0", "(Wojna polsko-ruska) is a 2009 Polish film directed by Xawery \u017bu\u0142awski based on the novel Polish-Russian War under the white-red flag by Dorota Mas\u0142owska."], ["1", "He is the son of actress Ma\u0142gorzata Braunek and director Andrzej \u017bu\u0142awski."]]}}
    :param file_path: 2wiki data file path, data/2wikimultihopqa/queries.jsonl
    :return:
    '''
    alias_dict = {}
    with open("data/2wikimultihopqa/id_aliases.json", 'r') as file:
        for line in file:
            '''
            example: {"Q_id": "Q3882501", "aliases": ["One Law for the Woman"], "demonyms": []}
            '''
            id_alias = json.loads(line)
            q_id = id_alias['Q_id']
            aliases = id_alias['aliases']
            alias_dict[q_id] = aliases

    with open("data/2wikimultihopqa/queries.jsonl", 'r') as file:
        count = 0
        examples = []

        for line in file:
            example = json.loads(line)
            examples.append(example)

        if split == 'train':
            start = 0
        else:
            start = 0+sample_number
        new_examples = []
        for example in examples[start: start+sample_number]:
            question = example['text']
            answer = example['metadata']['answer']
            answer_id = example['metadata']['answer_id']
            if answer_id not in alias_dict:
                aliases = []
            else:
                aliases = alias_dict[answer_id]
            answers = [answer] + aliases
            ctxs = [e[1] for e in example['metadata']['ctxs']]
            gt_ctxs = ctxs.copy()
            retrieval_ctx = search_documents(question, index_name="2wikimultihopqa", size=retrieve_number)
            for ctx in retrieval_ctx:
                ctx = ctx['_source']['text']
                if ctx not in ctxs:
                    ctxs.append(ctx)
            random.shuffle(ctxs)

            new_examples.append({
                'question': question,
                'answers': answers,
                'ctxs_candidate': ctxs,
                'ctxs_gt': gt_ctxs,
            })
            count = count + 1
            if count >= sample_number:
                break

    if split == 'test':
        file_name = f'data/2wikimultihopqa/tts_test_2_samples{sample_number}.json'
    else:
        file_name = 'data/2wikimultihopqa/tts_test.json'

    with open(file_name, 'w') as f:
        f.write(json.dumps(new_examples, indent=2))
        print(f'generate 2wikimultihopqa {file_name} done.')


    return examples


def generate_hotpotqa_retrieve_data(sample_number=100, retrieve_number=100, split='test'):
    '''
    '''

    with open("data/HotpotQA/hotpotqa-dev.json", 'r') as file:
        count = 0
        examples = []
        data = json.load(file)
        if split == 'train':
            start = 0
        else:
            start = 0+sample_number
        for example in data[start:start+sample_number]:
            question = example['question']
            answer = example['answer']
            context = example['context']
            context_dict = {}
            for e in context:
                context_dict[e[0]] = " ".join(e[1])
            supporting_facts = example['supporting_facts']
            gt_ctxs = []
            for e in supporting_facts:
                if context_dict[e[0]] not in gt_ctxs:
                    gt_ctxs.append(context_dict[e[0]])
            ctxs = gt_ctxs.copy()
            retrieval_ctx = search_documents(question, index_name="hotpotqa", size=retrieve_number)
            for ctx in retrieval_ctx:
                ctx = ctx['_source']['text']
                if ctx not in ctxs:
                    ctxs.append(ctx)
            random.shuffle(ctxs)
            ret_answers = []
            for ctx in gt_ctxs:
                i = ctxs.index(ctx)
                ret_answers.append(i)

            examples.append({
                'question': question,
                'answer': answer,
                'ret_answers': ret_answers,
                'ctxs_candidate': ctxs,
                'ctxs_gt': gt_ctxs,
            })
            count = count + 1
            if count >= sample_number:
                break

    if split == 'test':
        file_name = f'data/HotpotQA/tts_test_2_samples{sample_number}.json'
    else:
        file_name = 'data/HotpotQA/tts_train.json'

    with open(file_name, 'w') as f:
        f.write(json.dumps(examples, indent=2))
        print(f'generate hotpotqa {file_name} done.')


    return examples



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrieve data from Elasticsearch')
    parser.add_argument('--index_name', type=str, help='Index name')
    parser.add_argument('--file_path', type=str, help='File path')
    parser.add_argument('--query', type=str, help='Search query')
    parser.add_argument('--size', type=int, default=10, help='Search size')
    parser.add_argument('--op', type=str, help='operation.')
    parser.add_argument('--test_or_train', type=str, default='train', help='test or train.')
    parser.add_argument('--num_test_samples', type=int, default=100, help='')

    args = parser.parse_args()

    if args.op == 'index':
        file_path = args.file_path
        index_name = args.index_name
        build_index(index_name, file_path)
        print('build index done.')
    elif args.op == 'search':
        query = args.query
        index_name = args.index_name
        size = args.size
        results = search_documents(query, index_name, size)
        print('returned results:', results)
    elif args.op == 'generate_2wiki_data':
        if args.test_or_train == 'train':
            examples = generate_2wiki_retrieve_data(sample_number=100, split='train')
        elif args.test_or_train == 'test':
            examples = generate_2wiki_retrieve_data(sample_number=args.num_test_samples, split='test')
        print('example:', examples[0])
    elif args.op == 'generate_hotpotqa_data':
        if args.test_or_train == 'train':
            examples = generate_hotpotqa_retrieve_data(sample_number=100)
        elif args.test_or_train == 'test':
            examples = generate_hotpotqa_retrieve_data(sample_number=args.num_test_samples, split='test')

'''
python retriever.py --op index --index_name 2wikimultihopqa --file_path data/2wikimultihopqa/corpus.jsonl
python retriever.py --op index --index_name hotpotqa --file_path data/hotpotqa/corpus.jsonl
python retriever.py --op search --index_name 2wikimultihopqa --size 1 --query "Who is the mother of the director of film Polish-Russian War (Film)"
python retriever.py --op generate_hotpotqa_data
python retriever.py --op generate_2wiki_data --test_or_train test --num_test_samples 500
python retriever.py --op generate_hotpotqa_data --test_or_train test --num_test_samples 500
'''



