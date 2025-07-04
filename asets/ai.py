from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import os
import logging
import string
from googletrans import Translator

import dict

model_name = "cointegrated/rubert-tiny"



translator = Translator()
conf = translator.detect()
#result = translator.translate('' , src=conf.lang, dest='ru')

def scan_oscorb_message(message:str):
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    if message in list(string.ascii_lowercase):
        message=''.join(dict.translit_ru.get(c, c) for c in message)
    # Пример предсказания
    inputs = tokenizer(message, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)
    return True,predictions.item() if predictions.item() == 1 else False,predictions.item()
scan_oscorb_message('')
