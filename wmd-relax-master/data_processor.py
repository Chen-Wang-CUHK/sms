import json
import os


def collect_articles_summs(scores_json=None, summs_json=None, article_json=None, out_file=None, add_ref=False):
    if scores_json is None:
        scores_json = os.path.join('data', 'cnndm', 'raw_data', 'preprocess_lqual_scores_out.jsonl')
    if summs_json is None:
        summs_json = os.path.join('data', 'cnndm', 'raw_data', 'sys_summs_lqual_all.jsonl')
    if article_json is None:
        article_json = os.path.join('data', 'cnndm', 'raw_data', 'articles.jsonl')

    # read sys summs scores
    # we read the scores first since its ids was filtered
    summs_scores = {}
    with open(scores_json, encoding='utf-8') as scores_fr:
        for line in scores_fr:
            """
            {
            'id': str,
            'system': 'reference'/'seq2seq'/'pointer'/'ml'/'ml+rl',
            'annotators': list of 5 digits,
            'prompts': {
                        'hter': {'gold': ., 'human':[...], 'bleu-2': ., ...}, # gold is the average of human scores
                        'overall': {'gold': ., 'human':[...], 'bleu-2': ., ...},
                        'grammar': {'gold': ., 'human':[...], 'bleu-2': ., ...},
                        'redundancy': {'gold': ., 'human':[...], 'bleu-2': ., ...}
                        }
            }
            """
            line = line.strip()
            one_score = json.loads(line)
            id = one_score['id']
            sys_name = one_score['system']
            one_human_scores = {}
            if not add_ref and sys_name == 'reference':
                continue
            for sc_type in ['overall', 'grammar', 'redundancy']:
                one_human_scores[sc_type] = one_score['prompts'][sc_type]['gold']
            if id not in summs_scores:
                summs_scores[id] = {}
            summs_scores[id][sys_name] = one_human_scores

    # read sys summs
    sys_summs = {}
    with open(summs_json, encoding='utf-8') as summs_fr:
        for line in summs_fr:
            """
            {
            'id': str,
            'reference': str,
            'system': 'reference'/'seq2seq'/'pointer'/'ml'/'ml+rl',
            'text': str
            }
            """
            line = line.strip()
            one_summ = json.loads(line)
            id = one_summ['id']
            sys_name = one_summ['system']
            summ_text = one_summ['text']
            if not add_ref and sys_name == 'reference':
                continue
            if id in summs_scores and sys_name in summs_scores[id]:
                if id not in sys_summs:
                    sys_summs[id] = {}
                sys_summs[id][sys_name] = summ_text

    # read articles
    articles = {}
    with open(article_json, encoding='utf-8') as article_fr:
        for line in article_fr:
            """
            {
            'text': str,
            'id': str,
            'system': 'article'
            }
            """
            line = line.strip()
            one_article = json.loads(line)
            assert one_article['system'] == 'article'
            id = one_article['id']
            text = one_article['text']
            if id not in summs_scores:
                continue
            articles[id] = text

    # remove the data with only one system summaries and collect all data together
    filtered_all_data = {}
    summs_num = 0
    if out_file is not None:
        out_fw = open(out_file, 'w', encoding='utf-8')
    else:
        out_fw = None
    for id in summs_scores:
        if len(summs_scores[id]) < 4:
            continue
        filtered_all_data[id] = {'id': id, 'article': articles[id], 'sys_summs': sys_summs[id], 'summs_scores': summs_scores[id]}
        summs_num += len(sys_summs[id])
        saved_str = json.dumps(filtered_all_data[id])
        if out_fw is not None:
            out_fw.write(saved_str + '\n')
    print('Processing Finished. Num of instance: {}, Num of total Summs: {}'.format(len(filtered_all_data), summs_num))
    return filtered_all_data


if __name__ == '__main__':
    article_json = os.path.join('data', 'cnndm', 'raw_data', 'articles.jsonl')
    summs_json = os.path.join('data', 'cnndm', 'raw_data', 'sys_summs_lqual_all.jsonl')
    scores_json = os.path.join('data', 'cnndm', 'raw_data', 'preprocess_lqual_scores_out.jsonl')
    out_json = os.path.join('data', 'cnndm', 'cnndm_merged_filtered.jsonl')
    collect_articles_summs(scores_json=scores_json, summs_json=summs_json, article_json=article_json, out_file=out_json)