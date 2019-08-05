#!/usr/bin/env python
import argparse
import json
import csv
path_to_data = '/home/mr-mister/Ivan/PLN_ECI2019/TP/TP_PNL'
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sentences')
    ap.add_argument('labels', nargs='?')

    args = ap.parse_args()

    sentence_data = open(args.sentences, 'r')
    with open('test.txt','w') as f:
        if args.labels:
                label_data = open(args.labels, 'r')
                for sentence, captionID, label in zip(it_sentences(sentence_data), it_captionId(sentence_data), it_labels(label_data)):
                # Tenemos la oración en sentence con su categoría en label
                        f.write(sentence + '|' + captionID + '|' +label +'\n')
                        pass
        else:
                for sentence, captionID in zip(it_sentences(sentence_data), it_captionId(sentence_data)):
                # Tenemos una oración en sentence
                        f.write(sentence + '|' + captionID + '\n')
                #print(sentence)
                        pass
    
def it_sentences(sentence_data):
    for line in sentence_data:
        example = json.loads(line)
        yield example['sentence2']

def it_captionId(sentence_data):
    for line in sentence_data:
        example = json.loads(line)
        yield example['captionID']

def it_labels(label_data):
    label_data_reader = csv.DictReader(label_data)
    for example in label_data_reader:
        yield example['gold_label']




main()
