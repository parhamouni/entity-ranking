import pandas as pd
import gzip
import csv
from tqdm import tqdm
import gzip
import json
import os
import wget
import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
import subprocess
import bz2
import numpy as np
from wikimapper import WikiMapper
import pdb
warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)


class DataProcessing():
    def __init__(self, df_dbpedia_path = '../data/processed/dbpedia_201510.parquet.gz',
                        dbpedia_path = '../data/short_abstracts_en.tql.bz2', 
                        lucene_input_path = '../data/input/dbpedia201510.jsonl',
                        lucene_index_path = '../data/index',
                        topics_path = '../data/queries.tsv',
                        query_anserini_output =  '../data/run_anserini.sample.txt',
                        k = 1000,
                        qrel_path = '/home/parham/entity-ranking/data/DBpedia-Entity/collection/v2/qrels-v2.txt',
                        query_path = '../data/run_anserini.sample.txt',
                        query_index_path = '../data/DBpedia-Entity/collection/v2/queries-v2.txt',
                        indexed_data_save_address = '../data/processed/df_query_201510_bm25_k1000.parquet.gz',
                        wikidata_translation_v1_names = '../data/wikidata_translation_v1_names.json.gz',
                        wikidata_translation_v1_entities = '../data/wikidata_translation_v1.tsv.gz',
                        wikimapper_path = '../data/index_enwiki-20190420.db',
                        biggraph_entity_subset_path = '../data/processed/df_query_201510_bm25_k1000_biggraph_merge.parquet.gz',
                        deepct_path = '/home/negar/entity_deepct/DeepCT/reweighted_parham2/test_results.tsv',
                        deepct_weights='../data/processed/deep_ct_weights.parquet.gz',
                        deepct_tokens = '../data/processed/deep_ct_tokens.parquet.gz',
                        fold_address ='/home/parham/entity-ranking/data/DBpedia-Entity/collection/v2/folds/all_queries.json',
                        train_test_directory =  '../data/processed/folds/'):

        self.df_dbpedia_path = df_dbpedia_path
        self.dbpedia_path = dbpedia_path
        self.lucene_input_path = lucene_input_path
        self.lucene_index_path = lucene_index_path
        self.topics_path = topics_path
        self.query_anserini_output = query_anserini_output
        self.k = k 
        self.qrel_path = qrel_path
        self.query_path = query_path
        self.query_index_path = query_index_path
        self.indexed_data_save_address = indexed_data_save_address
        self.df_dbpedia_process=None
        self.wikidata_translation_v1_names = wikidata_translation_v1_names
        self.wikidata_translation_v1_entities = wikidata_translation_v1_entities
        self.wikimapper_path = wikimapper_path
        self.biggraph_entity_subset_path= biggraph_entity_subset_path
        self.deepct_path = deepct_path
        self.deepct_weights = deepct_weights
        self.deepct_token = deepct_tokens
        self.fold_address = fold_address
        self.train_test_directory = train_test_directory

    @staticmethod
    def flatten(sample):
        s =  sample.split('> ')
        results = []
        for i in s:
            results.extend(i.split('<'))
        results = [i for i in results]
        return results


    def load_dbpedia_corpus(self):
        try:
            df_dbpedia_process = pd.read_parquet(path = self.df_dbpedia_path)
        except: ## TODO add exception error

            #### load dbpedia
            if not os.path.exists(self.dbpedia_path):
                url = 'http://downloads.dbpedia.org/2015-10/core-i18n/en/short_abstracts_en.tql.bz2'
                filename = wget.download(url, out=os.path.dirname(self.dbpedia_path))
            ## read dbpedia
            with bz2.open(self.dbpedia_path, "r") as f:
                data = f.readlines()
                df_dbpedia = pd.DataFrame(data[1:-1])
                
            df_dbpedia[0] = df_dbpedia[0].apply(lambda x:x.decode("utf-8", "ignore"))
            df_dbpedia_process = df_dbpedia[0].apply(flatten)
            df_dbpedia_process = df_dbpedia_process.to_frame()
            df_dbpedia_process[1], df_dbpedia_process[2] ,df_dbpedia_process[3] , df_dbpedia_process[4], df_dbpedia_process[5], df_dbpedia_process[6], df_dbpedia_process[7]= zip(*list(df_dbpedia_process[0].values))
            df_dbpedia_process = df_dbpedia_process.drop(columns = [0,1,3,7])
            df_dbpedia_process.columns = ['url_dbpedia','predicate','abstract','url_wikipedia']
            df_dbpedia_process.to_parquet(self.df_dbpedia_path,compression='gzip')
        return df_dbpedia_process


    def lucence_index(self):
        # check if exists
        if not os.path.exists(self.lucene_index_path):
            os.makedirs(self.lucene_index_path)

        if len(os.listdir(self.lucene_index_path))==0:
            ## check input
            if not os.path.exists(self.lucene_input_path):
                self.df_dbpedia_process = self.load_dbpedia_corpus()
                df_dbpedia_for_lucene= self.df_dbpedia_process[['url_dbpedia','abstract']]
                df_dbpedia_for_lucene.rename(columns={'url_dbpedia':'id',
                                        'abstract':'contents'},inplace=True)
                df_dbpedia_for_lucene.to_json(self.lucene_input_path,
                                orient='records',
                                lines=True)


            command = [ 'python',
                        '-m',
                        'pyserini.index.lucene',
                        '--collection', 
                        'JsonCollection', 
                        '--input',
                        '{}'.format(os.path.dirname(self.lucene_input_path)),
                        '--index',
                        '{}'.format(self.lucene_index_path),
                        '--generator',
                        'DefaultLuceneDocumentGenerator'
                        '--threads',
                        '100' ,
                        '--storePositions',
                        '--storeDocvectors', 
                        '--storeRaw', 
                        '--bm25.accurate']
            process = subprocess.Popen(command, 
                                stdout=subprocess.PIPE,
                                universal_newlines=True)


    def query_command(self):
        command = [ 'python',
                    '-m',
                    'pyserini.search.lucene',
                    '--index',
                    '{}'.format(self.lucene_index_path),
                    '--topics', 
                    '{}'.format(self.topics_path),
                    '--outpsut', 
                    '{}'.format(self.query_anserini_output),
                    '--bm25',
                    '--hits',
                    '{}'.format(self.k)]
        process = subprocess.Popen(command, 
                            stdout=subprocess.PIPE,
                            universal_newlines=True)


    def query_merge_process(self):
        try: 
            df_query_results_merge = pd.read_parquet(self.indexed_data_save_address)
        except:
            df_query_results = pd.read_csv(self.query_path,
                                sep='\s',
                                header = None)
            df_query_results.drop(columns=[1,5],inplace=True)
            df_query_results.rename(columns={
                0:'query_index',
                2: 'id',
                3:'hit',
                4 :'score'
            },inplace=True)
            df_query_results.rename(columns = {'id':'url_dbpedia'},inplace=True)
            if not self.df_dbpedia_process:
                self.df_dbpedia_process = self.load_dbpedia_corpus()
            df_query_results_merge= df_query_results.merge(self.df_dbpedia_process,on = 'url_dbpedia')
            df_qrel = pd.read_csv(self.qrel_path,delimiter='\t',header=None)
            df_qrel.columns=['query_id',1,'dbpedia_entity','qrel']
            df_qrel.drop(columns=1,inplace=True)
            df_query_index = pd.read_csv(self.query_index_path,delimiter='\t',header=None)
            df_query_index.reset_index(inplace=True)
            df_query_index.rename(columns={
                'index':'query_index',
                0:'query_id',
                1:'query_text'
                },inplace=True)
            df_qrel.rename(columns = 
                    {'query_index':'query_id'},inplace=True)
            df_query_index = df_query_index.merge(df_qrel, on = 'query_id')
            df_query_results_merge['dbpedia_entity'] = df_query_results_merge.url_dbpedia.apply(lambda x: '<dbpedia:' + x.split('resource/')[-1]+'>')
            df_query_results_merge = df_query_results_merge.merge(df_query_index[['query_index','query_id','query_text']].drop_duplicates(), on = ['query_index'])
            df_query_results_merge =df_query_results_merge.merge(df_query_index[['query_index','dbpedia_entity','qrel']], on = ['query_index','dbpedia_entity'], how='left')
            df_query_results_merge.sort_values(by = ['query_index','hit'],inplace=True)
            df_query_results_merge.reset_index(inplace=True, drop = True)
            df_query_results_merge.to_parquet(self.indexed_data_save_address,compression='gzip')
        return df_query_results_merge

    def biggraph_data(self): 
        ## todo wget
        try: 
            df_wiki2016_merge_biggraph_emb = pd.read_parquet(self.biggraph_entity_subset_path)
            df_wiki2016_merge_biggraph_emb.sort_values(by = ['query_index','hit'],inplace=True)
            df_wiki2016_merge_biggraph_emb.reset_index(inplace=True, drop = True)
        except: ## TODO except reason

            with gzip.open(self.wikidata_translation_v1_names, "rt") as f:
                expected_dict = json.load(f)
            df_entities  = pd.DataFrame(expected_dict)
            df_entities.rename(columns = {'0':'entity'},inplace=True)

            df_wiki2016 = self.query_merge_process()

            ## TODO: wget wikimapper
            mapper = WikiMapper(self.wikimapper_path)
            df_wiki2016['title_to_id_wikidata'] = df_wiki2016.dbpedia_entity.apply(lambda x:mapper.title_to_id(x[9:-1].replace(' ','_')))
            df_wiki2016['entity'] = '<http://www.wikidata.org/entity/' + df_wiki2016.title_to_id_wikidata + '>'
            df_entities.reset_index(inplace=True)
            df_entities.rename(columns = {
                'index':'entity_index',0:'entity'
            },inplace=True)

            df_wiki2016_merge = df_wiki2016.merge(df_entities, on = 'entity', how = 'left')
            df_wiki2016_merge.sort_values(by = 'entity_index',inplace=True)
            entity_valid_indices = df_wiki2016_merge.entity_index.dropna().drop_duplicates().values

            
            rows = []
            with gzip.open(self.wikidata_translation_v1_entities, 'rt') as f:
                tsv_reader = csv.reader(f, delimiter="\t")
                print(tsv_reader.line_num)
                number_of_lines = int(entity_valid_indices.max())
                for i in tqdm(range(number_of_lines)):
                    row = next(tsv_reader)
                    if i-1 in entity_valid_indices:
                        rows.append(row)

            df_en = pd.DataFrame(rows)
            df_en.columns = [str(s) for s in df_en.columns]
            df_biggraph_embeddings = df_en[df_en.columns[1:]].apply(lambda x: list(x.values),axis=1) 
            df_biggraph_entities = df_en['0']
            df_biggraph_embedding_entity = pd.concat([df_biggraph_entities,df_biggraph_embeddings],axis = 1)
            df_biggraph_embedding_entity.columns = ['entity','biggraph_embedding']
            df_wiki2016_merge_biggraph_emb=df_wiki2016_merge.merge(df_biggraph_embedding_entity,on = 'entity', how = 'left')
            df_wiki2016_merge_biggraph_emb.sort_values(by = ['query_index','hit'],inplace=True)
            df_wiki2016_merge_biggraph_emb.reset_index(inplace=True, drop = True)
            df_wiki2016_merge_biggraph_emb.to_parquet(self.biggraph_entity_subset_path,
                                                        compression = 'gzip')
        return df_wiki2016_merge_biggraph_emb


    def deep_ct_process(self): 
        try:
            df_weights = pd.read_parquet(self.deepct_weights)
            df_tokens = pd.read_parquet(self.deepct_token)
        except:
            df_ct = pd.read_csv(self.deepct_path,delimiter='\s', header = None,index_col = False)
            tokens_columns = [f for f in df_ct.columns if f%2==0]
            weight_columns = [f for f in df_ct.columns if f%2==1]
            dfc_tokens = df_ct[tokens_columns]
            df_weights = df_ct[weight_columns]
            df_tokens = dfc_tokens.apply(lambda x: list(x.values),axis=1) 
            df_weights = df_weights.apply(lambda x: list(x.values),axis=1) 
            df_weights = df_weights.to_frame()
            df_weights.columns =  [str(s) for s in df_weights.columns]
            df_tokens = df_tokens.to_frame()
            df_tokens.columns =  [str(s) for s in df_tokens.columns]
            df_weights.to_parquet(self.deepct_weights,compression='gzip')
            df_tokens.to_parquet(self.deepct_token,compression='gzip')
        return df_weights, df_tokens

    def triplet_data(self):
        df_folds =pd.read_json(self.fold_address)
        df_folds = df_folds.T
        df_folds.reset_index(inplace = True)
        df_folds.rename(columns={'index':'fold'},inplace=True)
        df_wiki2016_merge_biggraph_emb = self.biggraph_data()
        df_wiki2016_merge_biggraph_emb.biggraph_embedding = df_wiki2016_merge_biggraph_emb.biggraph_embedding.fillna(df_wiki2016_merge_biggraph_emb.biggraph_embedding.notna().apply(lambda x: x or ['0.0']*200))
        df_wiki2016_merge_biggraph_emb.biggraph_embedding= df_wiki2016_merge_biggraph_emb.biggraph_embedding.apply(lambda x:list(map(float, x)))
        df_wiki2016_merge_biggraph_emb.biggraph_embedding=df_wiki2016_merge_biggraph_emb.biggraph_embedding.apply(np.array)
        df_weights, df_tokens = self.deep_ct_process()
        df_weights.rename(columns={'0':'deepct_weights'},inplace=True)
        df_tokens.rename(columns={'0':'deepct_tokens'},inplace=True)
        df_wiki2016_merge_biggraph_emb = df_wiki2016_merge_biggraph_emb.join(df_tokens.join(df_weights))
        df_wiki2016_merge_biggraph_emb.abstract = df_wiki2016_merge_biggraph_emb.abstract.apply(lambda x:x[1:].split('"@en ')[0])
        for i in tqdm(range(1)):
            string = "training_set_fold_{} = df_wiki2016_merge_biggraph_emb[df_wiki2016_merge_biggraph_emb.query_id.isin(df_folds[df_folds.fold=={}].training.values[0])]".format(i,i)
            exec(string)
            string = "test_set_fold_{} = df_wiki2016_merge_biggraph_emb[df_wiki2016_merge_biggraph_emb.query_id.isin(df_folds[df_folds.fold=={}].testing.values[0])]".format(i,i)
            exec(string)
            for s in ['training','test']:
                # pdb.set_trace()
                string = "{}_set_fold_{}[['query_id','query_text','abstract','dbpedia_entity','qrel','biggraph_embedding','deepct_weights','deepct_tokens']]".format(s,i)
                ## deep_ct to be added
                df_sample = eval(string)
                positive_docs = df_sample.loc[df_sample['qrel']>0]
                negative_docs = df_sample.loc[df_sample['qrel']==0]
                df_positive_counts = positive_docs.groupby('query_id').query_text.count()
                negative_dfs = []
                for query_id, g in negative_docs.groupby('query_id'):
                    try:
                        size = df_positive_counts[df_positive_counts.index==query_id].values[0]
                        negative_dfs.append(g.sample(size))
                    except:
                        pass
                negative_docs=pd.concat(negative_dfs)
                result = pd.merge(positive_docs, negative_docs, on=['query_id','query_text'])
                string = \
                "result.to_parquet('{}fold_{}_{}.parquet.gzip',compression='gzip')".format(self.train_test_directory,i,s)
                exec(string)

if __name__=='__main__':
    dp = DataProcessing()
    dp.triplet_data()
    # dp.deep_ct_process()
