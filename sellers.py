import csv
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = 'sellers'
DIMENSION = 384
MILVUS_HOST = 'localhost'
MILVUS_PORT = '19530'
BATCH_SIZE = 128
COUNT = 500

connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
print('Connected to Milvus')

if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)

fields = [    
    FieldSchema(name='seller_id',  dtype=DataType.VARCHAR, is_primary=True, max_length=15),
    FieldSchema(name='seller_badge', dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name='seller_username', dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name='seller_country', dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name='seller_products_sold', dtype=DataType.DOUBLE),
    FieldSchema(name='seller_num_products_listed', dtype=DataType.DOUBLE),
    FieldSchema(name='seller_community_rank', dtype=DataType.DOUBLE),
    FieldSchema(name='seller_num_followers', dtype=DataType.DOUBLE),
    FieldSchema(name='seller_pass_rate', dtype=DataType.DOUBLE),
    FieldSchema(name='username_emb', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
]

schema = CollectionSchema(fields=fields)

collection = Collection(name=COLLECTION_NAME, schema=schema)
collection.create_index(field_name="username_emb", index_params={'metric_type': 'L2', 'index_type': 'IVF_FLAT', 'params': {'nlist': 1536}})
collection.load()
print('Collection created and indices created')

transformer = SentenceTransformer('all-MiniLM-L6-v2')

def csv_load(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        next(reader)  
        for row in reader:
            if '' in row:
                continue
            yield row

def embed_insert(data):
    seller_ids = []
    seller_badges = []
    seller_usernames = []
    seller_countries = []
    seller_products_solds = []
    seller_num_products_listeds = []
    seller_community_ranks = []
    seller_num_followerss = []
    seller_pass_rates = []
    username_embs = []

    for row in data.values():
        seller_badges.append(row[23])
        seller_ids.append(row[27])
        seller_usernames.append(row[28])
        seller_countries.append(row[30])
        seller_products_solds.append(float(row[31]))
        seller_num_products_listeds.append(float(row[32]))
        seller_community_ranks.append(float(row[33]))
        seller_num_followerss.append(float(row[34]))
        seller_pass_rates.append(float(row[35]))
        username_embs.append([x for x in transformer.encode(row[28])])

    collection.insert([
        seller_ids, seller_badges, seller_usernames, seller_countries,
        seller_products_solds, seller_num_products_listeds, seller_community_ranks,
        seller_num_followerss, seller_pass_rates, username_embs
    ])
count = 0
unique_sellers = {}
data_batch = []

try:
    for row in csv_load("data/vestiaire.csv"):
        seller_id = row[27]
        if seller_id not in unique_sellers:
            unique_sellers[seller_id] = row
            count += 1 

        if count >= COUNT:
            break

        if len(unique_sellers) % BATCH_SIZE == 0:
            embed_insert(unique_sellers)
            unique_sellers = {}

    if len(unique_sellers) != 0:
        embed_insert(unique_sellers)

    collection.flush()
    print('Inserted data successfully')
    print('Number of inserted items:', count)

    
except Exception as e:
    print('Error occurred during data insertion:', str(e))
    raise e
