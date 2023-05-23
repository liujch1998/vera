import os
import sys
import json
from tqdm import tqdm

import torch
import transformers
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from data import FORMAT_BY_TASK

task = sys.argv[1]
split = sys.argv[2]
assert task in FORMAT_BY_TASK
assert split in ['train', 'dev', 'test', 'test.all']

model_name = '../ckpt/question-converter-t5-3b'
if 'bart' in model_name:
    model_type = 'bart'
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)
elif 't5' in model_name:
    model_type = 't5'
    config = transformers.AutoConfig.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config)
model.to(device)
print('decoder start token id:', model.config.decoder_start_token_id)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def parse_choices(s):
    choices = []
    key = 'A' if s.find('(A)') != -1 else 'a'
    while True:
        pos = s.find(f'({chr(ord(key) + 1)})')
        if pos == -1:
            break
        choice = s[3:pos]
        s = s[pos:]
        choice = choice.strip(' ')
        choices.append(choice)
        key = chr(ord(key) + 1)
    choice = s[3:]
    choice = choice.strip(' ')
    choices.append(choice)
    return choices

def parse_line(line):
    q, a = line.strip('\n').split('\t')
    q = q.strip(' ')
    a = a.strip(' ')
    context = q.split('\\n')[2].strip(' ') if len(q.split('\\n')) > 2 else None
    question = q.split('\\n')[0].strip(' ')
    if question == '':
        question = None
    choices = parse_choices(q.split('\\n')[1].strip(' '))
    answer = a
    return context, question, choices, answer

def parse_line_bool(line):
    q, a = line.strip('\n').split('\t')
    q = q.strip(' ')
    a = a.strip(' ')
    if '\\n' not in q: # com2sense, spatial_cs, csqa2, strategyqa, creak
        if task == 'com2sense':
            question = None
            choice = q
        elif task == 'spatial_cs':
            question = q
            choice = 'yes'
        elif task == 'csqa2':
            question = q
            choice = 'yes'
        elif task == 'strategyqa':
            question = q
            choice = 'yes'
        elif task == 'creak':
            question = None
            choice = q
    else: # truthfulqa_mc2
        question = q.split('\\n')[0].strip(' ')
        choice = q.split('\\n')[1].strip(' ')
    label = a
    assert label in ['yes', 'no']
    return question, choice, label

def split_question(question):
    if question.find('. ') == -1:
        return None, question
    else:
        question_splitted = question.split('. ')
        prefix_sentences = '. '.join(question_splitted[:-1]) + '.'
        question = question_splitted[-1]
        return prefix_sentences, question

def qa2d(context, question, choice, task):
    if question is None: # no question, the choice is the declarative statement; applies to Winogrande and Com2Sense and CREAK
        d = choice
        return d
    
    if task in ['copa', 'swag', 'hellaswag', 'codah', 'story_cloze_test']:
        d = f'{question} {choice}'
        return d

    # if '<mask>' in question: # applies to NumerSense
    #     d = question.replace('<mask>', choice)
    #     return d

    # if '[MASK]' in question: # applies to PROST
    #     d = question.replace('[MASK]', choice)
    #     return d

    num_blanks = 0
    if len(question) >= 2 and (' _' in question or '_ ' in question or (question[0] == '_' and question[1].isalpha())): # cloze-style question
        # QASC has some questions like "_have the amazing capacity to regrow segments that break off."
        if question[0] == '_' and question[1].isalpha():
            question = f'_ {question[1:]}'
        d = question
        for i in range(20, 0, -1):
            # Assume that the blank has a space before it and/or after it
            num_blanks += d.count(' ' + '_' * i)
            d = d.replace(' ' + '_' * i, ' ' + choice)
            num_blanks += d.count('_' * i + ' ')
            d = d.replace('_' * i + ' ', choice + ' ')
        if num_blanks != 1:
            print('Number of blanks is not 1! Not treating as cloze-style question.')
            print(question)

    if num_blanks != 1:
        # if not question.endswith('?'): # continuation-style question
        #     d = f'{question} {choice}'
        # else:
        prefix_sentences, question = split_question(question)
        if 'bart' in model_name:
            s = f'question: {question} answer: {choice}'
        elif 't5' in model_name:
            s = f'{question} </s> {choice}'
        input_ids = tokenizer(s, return_tensors='pt').input_ids.to(device)
        output = model.generate(input_ids=input_ids, max_length=256)
        d = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        if prefix_sentences is not None:
            d = f'{prefix_sentences} {d}'

    if not (d.endswith('.') or d.endswith('!')):
        d = d.rstrip('?') + '.'
    if context is not None:
        d = f'{context} {d}'
    return d

print(f'Converting {task} ...')
path = f'../data/{task}/{split}.tsv'
with open(path) as f:
    lines = f.readlines()
ds = []
for line in tqdm(lines):
    if FORMAT_BY_TASK[task] == 'mc':
        try:
            context, question, choices, answer = parse_line(line)
        except:
            print(f'Failed to parse line! Skipping ...')
            print(line)
            continue
        golds = [qa2d(context, question, answer, task)]
        distractors = []
        for choice in choices:
            if choice == answer:
                continue
            distractors.append(qa2d(context, question, choice, task))
        if None in golds or None in distractors:
            print(f'One of the statements is None! Skipping ...')
            print(line)
            continue
        ds.append({ 'golds': golds, 'distractors': distractors })
    elif FORMAT_BY_TASK[task] == 'bool':
        try:
            question, choice, label = parse_line_bool(line)
        except:
            print(f'Failed to parse line! Skipping ...')
            print(line)
            continue
        golds, distractors = [], []
        if label == 'yes':
            golds.append(qa2d(None, question, choice, task))
        elif label == 'no':
            distractors.append(qa2d(None, question, choice, task))
        if None in golds or None in distractors:
            print(f'One of the statements is None! Skipping ...')
            print(line)
            continue
        ds.append({ 'golds': golds, 'distractors': distractors })

ensure_dir(f'../data/{task}')
with open(f'../data/{task}/{split}.json', 'w') as g:
    json.dump(ds, g, indent=4)
