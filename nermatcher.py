import spacy
import time
from collections import Counter
from spacy import displacy
from string import punctuation

start = time.time()

nlp = spacy.load('ru_core_news_sm', disable=['ner'])

config = {
   "phrase_matcher_attr": 'LEMMA',
   "validate": True,
   "overwrite_ents": False,
   "ent_id_sep": "||",
}
ruler = nlp.add_pipe("entity_ruler", config=config)

tlist = open('C:/Main/dicts/jurdict.txt', encoding='utf-8').read().splitlines()

patterns = []
for i in tlist:
   patterns.append({"label": "JURTERM", "pattern": i})
ruler.add_patterns(patterns)

doc = nlp(open('C:/Main/corpus/sp11b.txt', 'r', encoding='utf-8').read())

terminy = [(ent.text) for ent in doc.ents]
tokens = [token.text for token in doc if token.text not in punctuation]

print(len(terminy), len(tokens))

end = time.time()
print("Готов вкалывать", end-start)

displacy.serve(doc, style='ent')