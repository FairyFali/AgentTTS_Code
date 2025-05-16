import json

with open('hotpotqa-dev.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

corpus =[]
for item in data:
    context = item['context']
    question = item['question']
    answer = item['answer']
    supporting_facts= item['supporting_facts']

    for e in context:
        title = e[0]
        text = " ".join(e[1])
        corpus.append({"title": title, "text": text})

with open('corpus.jsonl', 'w', encoding='utf-8') as f:
    for item in corpus:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
