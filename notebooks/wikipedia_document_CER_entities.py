import pandas as  pd
import wikipedia
from tqdm import tqdm


df_entities = pd.read_json('../data/entities.json', lines=True)
nodes = set(df_entities.entities.sum())

results = []
for n in tqdm(nodes):
    try:
        document = wikipedia.page(n).content
    except:
        document = None
    results.append(
        {'entity':n,
        'document':document}
    )
df_results  = pd.DataFrame(results)
df_results.to_csv('entity_pages.csv')