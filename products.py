import csv
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = 'products'
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
    FieldSchema(name='product_id', dtype=DataType.VARCHAR, is_primary=True, max_length=15),
    FieldSchema(name='product_type', dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name='product_name', dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name='product_description', dtype=DataType.VARCHAR, max_length=3000),
    FieldSchema(name='product_gender_target', dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name='product_category', dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name='product_season', dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name='product_condition', dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name='product_like_count', dtype=DataType.DOUBLE),
    FieldSchema(name='sold', dtype=DataType.BOOL),
    FieldSchema(name='brand_id', dtype=DataType.VARCHAR, max_length=15),
    FieldSchema(name='brand_name', dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name='product_material', dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name='product_color', dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name='price_usd', dtype=DataType.DOUBLE),
    FieldSchema(name='name_emb', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
    FieldSchema(name='descr_emb', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
]

schema = CollectionSchema(fields=fields)

collection = Collection(name=COLLECTION_NAME, schema=schema)
collection.create_index(field_name="name_emb", index_params={'metric_type': 'L2', 'index_type': 'IVF_FLAT', 'params': {'nlist': 1536}})
collection.create_index(field_name="descr_emb", index_params={'metric_type': 'L2', 'index_type': 'IVF_FLAT', 'params': {'nlist': 1536}})
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

    product_ids = []
    product_types = []
    product_names = []
    product_descriptions = []
    product_gender_targets = []
    product_categories = []
    product_seasons = []
    product_conditions = []
    product_like_counts = []
    solds = []
    brand_ids = []
    brand_names = []
    product_materials = []
    product_colors = []
    price_usds = []
    name_embs = []
    descr_embs = []

    for row in data:
        product_ids.append(row[0])
        product_types.append(row[1])
        product_names.append(row[2])
        product_descriptions.append(row[3])
        product_gender_targets.append(row[5])
        product_categories.append(row[6])
        product_seasons.append(row[7])
        product_conditions.append(row[8])
        product_like_counts.append(float(row[9]))
        solds.append(bool(row[10]))
        brand_ids.append(row[15])
        brand_names.append(row[16])
        product_materials.append(row[18])
        product_colors.append(row[19])
        price_usds.append(float(row[20]))
        name_embs.append([x for x in transformer.encode(row[2])])
        descr_embs.append([x for x in transformer.encode(row[3])])

    collection.insert([
        product_ids, product_types, product_names, product_descriptions,
        product_gender_targets, product_categories, product_seasons, product_conditions,
        product_like_counts, solds, brand_ids, brand_names, product_materials,
        product_colors, price_usds, name_embs, descr_embs
    ])

count = 0
data_batch = []

try:
    for row in csv_load("data/vestiaire.csv"):
        if count <= COUNT:
            data_batch.append(row)

            if len(data_batch) % BATCH_SIZE == 0:
                embed_insert(data_batch)
                data_batch = []
            count += 1
        else:
            break

    if len(data_batch) != 0:
        embed_insert(data_batch)

    collection.flush()
    print('Inserted data successfully')
    print('Number of inserted items:', count)

except Exception as e:
    print('Error occurred during data insertion:', str(e))
    raise e
