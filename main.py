# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
import json

d = {}
i = 0
with open('snli_1.0_train_filtered.jsonl') as f:
    json_data = [json.loads(line) for line in f]
        
    

    