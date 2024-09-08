import csv
from fastapi import FastAPI, Path, Body, HTTPException
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility, MilvusClient
from sentence_transformers import SentenceTransformer
import numpy as np
from fastapi import Query
from fastapi.responses import JSONResponse
from py_eureka_client import eureka_client
from typing import List, Dict
import logging
import uvicorn
import os
import socket
import json

app = FastAPI()
APP_NAME = "vector-database-service"
INSTANCE_PORT = 8001


COLLECTION_NAME_PRODUCTS = 'products'
COLLECTION_NAME_SELLERS = 'sellers'
DIMENSION = 384
MILVUS_HOST = 'localhost'
MILVUS_PORT = 19530
BATCH_SIZE = 128
TOP_K = 5
COUNT = 10000

# Connect to Milvus
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
collection_products = Collection(name=COLLECTION_NAME_PRODUCTS)
collection_sellers = Collection(name=COLLECTION_NAME_SELLERS)



# Check connection to Milvus and count items inserted into collection
@app.get("/test-milvus-connection/{collection_name}")
async def test_milvus_connection(collection_name: str):
    try:
        client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}", token="root:Milvus")
        status = client.get_collection_stats(collection_name=collection_name)
        return {"message": f"Connected to Milvus collection '{collection_name}'", "status": status}
    except Exception as e:
        return {"message": "Error occurred during Milvus connection:", "error": str(e)}




# count entities in a collection
@app.get("/get-entity-count/{collection_name}")
async def get_entity_count(collection_name: str):
    try:
        client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}", token="root:Milvus")

        stats = client.get_collection_stats(collection_name=collection_name)
        count = stats['row_count'] 

        return {"message": f"Count of entities in collection '{collection_name}'", "count": count}
    except Exception as e:
        return {"message": "Error occurred while getting entity count:", "error": str(e)}



# delete product by id

@app.delete("/collections/{collection_name}/delete_product/{product_id}")
async def delete_product_by_id(
    collection_name: str = Path(..., title="The name of the Milvus collection"),
    product_id: int = Path(..., title="ID of the product to delete")
):
    try:
        client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}", token="root:Milvus")

        response = client.delete(collection_name=collection_name, pks=f'{product_id}')
        print(response)
        if response:
            return JSONResponse(content={"message": f"Product with ID {product_id} deleted successfully"})
        else:
            return JSONResponse(content={"message": f"Product with ID {product_id} not found"}, status_code=404)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    


# delete seller by id

@app.delete("/collections/{collection_name}/delete_seller/{seller_id}")
async def delete_seller_by_id(
    collection_name: str = Path(..., title="The name of the Milvus collection"),
    seller_id: int = Path(..., title="ID of the seller to delete")
):
    try:
        client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}", token="root:Milvus")


        response = client.delete(collection_name=collection_name, pks=f'{seller_id}')
        print(response)
        if response:
            return JSONResponse(content={"message": f"Seller with ID {seller_id} deleted successfully"})
        else:
            return JSONResponse(content={"message": f"Seller with ID {seller_id} not found"}, status_code=404)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Load SentenceTransformer model
transformer = SentenceTransformer('all-MiniLM-L6-v2')


# get product by id

@app.get("/collections/{collection_name}/getproductby_id/{product_id}")
async def get_by_product_id(
    collection_name: str,
    product_id: int
):
    try:
        client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}", token="root:Milvus")
        
        res = client.query(
            collection_name=collection_name,
            filter=f"product_id LIKE '{product_id}'",
            output_fields = [
                "product_id", "product_type", "product_name", "product_description", "product_gender_target",
                "product_category", "product_season", "product_condition", "product_like_count", "sold",
                "brand_id", "brand_name", "product_material", "product_color", "price_usd"
            ],
            consistency_level="Strong"
        )
        
        return {"query_result": res}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# get seller by id

@app.get("/collections/{collection_name}/getseller_by_id/{seller_id}")
async def get_by_seller_id(
    collection_name: str,
    seller_id: int
):
    try:
        client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}", token="root:Milvus")

        
        res = client.query(
            collection_name=collection_name,
            filter=f"seller_id LIKE '{seller_id}'",
            output_fields = [
                "seller_id", "seller_badge", "seller_username", "seller_country",
                "seller_products_sold", "seller_num_products_listed", "seller_community_rank",
                "seller_num_followers", "seller_pass_rate"
            ],
            consistency_level="Strong"
        )
        
        return {"query_result": res}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
       


def generate_embedding(text):
    embeddings = model.encode(text)
    return embeddings.tolist()


# add new product to collection

@app.post("/collections/{collection_name}/add_product")
async def add_product(
    collection_name: str = Path(..., description="Name of the Milvus collection"),
    product_data: dict = Body(..., description="Data of the product to be added")
):
    try:
        client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}", token="root:Milvus")

        if 'product_name' in product_data:
            product_data['name_emb'] = generate_embedding(product_data['product_name'])
        if 'product_description' in product_data:
            product_data['descr_emb'] = generate_embedding(product_data['product_description'])        
       
        res = client.insert(collection_name=collection_name, data=[product_data])
       
        if res:
            return JSONResponse(content={"message": "Product added successfully"})
        else:
            return JSONResponse(content={"message": "Error while inserting new product"})
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)



# add new seller to collection

@app.post("/collections/{collection_name}/add_seller")
async def add_seller(
    collection_name: str = Path(..., description="Name of the Milvus collection"),
    seller_data: dict = Body(..., description="Data of the seller to be added")
):
    try:
        client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}", token="root:Milvus")

        if 'seller_username' in seller_data:
            seller_data['username_emb'] = generate_embedding(seller_data['seller_username'])
       
        res = client.insert(collection_name=collection_name, data=[seller_data])
       
        if res:
            return JSONResponse(content={"message": "Seller added successfully"})
        else:
            return JSONResponse(content={"message": "Error while inserting new seller"})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)



# update or insert product

@app.post("/collections/{collection_name}/upsert_product")
async def upsert_product(
    collection_name: str = Path(..., description="Name of the Milvus collection"),
    product_data: dict = Body(..., description="Data of the product to be upserted")
):
    try:
        client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}", token="root:Milvus")

        if 'product_name' in product_data:
            product_data['name_emb'] = generate_embedding(product_data['product_name'])
        if 'product_description' in product_data:
            product_data['descr_emb'] = generate_embedding(product_data['product_description'])          
       
        res = client.upsert(collection_name=collection_name, data=[product_data])

        if res:
            return JSONResponse(content={"message": "Product upserted successfully"})
        else:
            return JSONResponse(content={"message": "Error while upserting new product"})
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    


# update or insert seller

@app.post("/collections/{collection_name}/upsert_seller")
async def upsert_seller(
    collection_name: str = Path(..., description="Name of the Milvus collection"),
    seller_data: dict = Body(..., description="Data of the seller to be upserted")
):
    try:
        client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}", token="root:Milvus")

        if 'seller_username' in seller_data:
            seller_data['username_emb'] = generate_embedding(seller_data['seller_username'])    

        res = client.upsert(collection_name=collection_name, data=[seller_data])

        if res:
            return JSONResponse(content={"message": "Seller upserted successfully"})
        else:
            return JSONResponse(content={"message": "Error while upserting new seller"})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# query 1
# Prebroj zenske proizvode brenda Gucci

@app.get("/query1/{collection_name}")
async def count_products(collection_name: str = Path(..., description="Name of the Milvus collection")):
    try:
        collection = Collection(name=collection_name)

        expr = f'{"brand_name"} == "Gucci" && {"product_gender_target"} == "Women"'

        counts = collection.query(expr=expr, output_fields=["count(*)"])
        count_result = counts[0]['count(*)']  

        return {"count": count_result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying Milvus collection '{collection_name}': {str(e)}")



# search with embedding function for query 2

def search_sellers_with_embedding(search_term: str) -> List[Dict[str, str]]:
    try:
        logging.basicConfig(level=logging.INFO)
        embedding = model.encode([search_term])
        print("Generated embedding:", embedding)

        search_params = {
            "metric_type": "L2"
        }
        client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}", token="root:Milvus")
        print("Milvus client connected:", client)

        results = client.search(
            collection_name='sellers',
            data=embedding,
            anns_field="username_emb",
            search_params=search_params, 
            limit=10,
            output_fields = [
                "seller_badge", "seller_username", "seller_country",
                "seller_num_followers", "seller_pass_rate"
            ],
        )

        search_results = []

        for hit in results[0]:
            
            seller_badge = hit['entity']['seller_badge']
            seller_username = hit['entity']['seller_username']
            seller_country = hit['entity']['seller_country']
            seller_num_followers = hit['entity']['seller_num_followers']
            seller_pass_rate = hit['entity']['seller_pass_rate']


            search_results.append({
                "seller_badge": seller_badge,
                "seller_username": seller_username,
                "seller_country": seller_country,
                "seller_num_followers": seller_num_followers,
                "seller_pass_rate": seller_pass_rate
            })

            print("Parsed result:", {
                "seller_badge": seller_badge,
                "seller_username": seller_username,
                "seller_country": seller_country,
                "seller_num_followers": seller_num_followers,
                "seller_pass_rate": seller_pass_rate
            })
        
        return search_results
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return [{"error": str(e)}]


# query 2
# Pronadji 10 prodavaca koji imaju najslicniji username unetom

@app.get("/query2")
async def perform_search(search_term: str = Query(..., title="Search Term", description="Term to search for")):
    search_results = search_sellers_with_embedding(search_term)
    print("Received search term:", search_term)
    print("Search results:", search_results)

    return {"search_results": search_results}




model = SentenceTransformer('all-MiniLM-L6-v2')


# search with embedding function for complex query 1

def search_with_embedding(search_term: str, collection_name: str, price_lower: float, price_upper: float):
    try:

        logging.basicConfig(level=logging.INFO)
        embedding = model.encode([search_term])
        print("Generated embedding:", embedding)

        filter_expr = f'product_condition == "Never worn" && price_usd >= {price_lower} && price_usd <= {price_upper}'

        search_params = {"metric_type": "L2"}

        client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}", token="root:Milvus")
        print("Milvus client connected:", client)

        results = client.search(
            collection_name=collection_name,
            data=embedding,
            anns_field="descr_emb",
            search_params=search_params,
            limit=10,
            filter=filter_expr,
            output_fields = [
                "product_type", "product_name", "product_description", "product_gender_target",
                "product_category", "product_condition", "product_like_count", "brand_name", "price_usd"
            ],
        )

        search_results = []

        for hit in results[0]:
            
            product_type = hit['entity']['product_type']
            product_name = hit['entity']['product_name']
            product_description = hit['entity']['product_description']
            product_gender_target = hit['entity']['product_gender_target']
            product_category = hit['entity']['product_category']
            product_condition = hit['entity']['product_condition']
            product_like_count = hit['entity']['product_like_count']
            brand_name = hit['entity']['brand_name']
            price_usd = hit['entity']['price_usd']

            search_results.append({
                "product_type": product_type,
                "product_name": product_name,
                "product_description": product_description,
                "product_gender_target": product_gender_target,
                "product_category": product_category,
                "product_condition": product_condition,
                "product_like_count": product_like_count,
                "brand_name": brand_name,
                "price_usd": price_usd
            })

            print("Parsed result:", {
                "product_type": product_type,
                "product_name": product_name,
                "product_description": product_description,
                "product_gender_target": product_gender_target,
                "product_category": product_category,
                "product_condition": product_condition,
                "product_like_count": product_like_count,
                "brand_name": brand_name,
                "price_usd": price_usd
            })
        
        return search_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying Milvus collection '{collection_name}': {str(e)}")




# complex query 1
# Pronadji 10 najslicnijih proizvoda po unetom opisu. Takodje, filtriraj proizvode koji nisu nikad noseni
# na osnovu unetih vrednosti za opseg cene

@app.get("/complex_query_1")
async def perform_search(
    search_term: str = Query(..., title="Search Term", description="Term to search for"),
    collection_name: str = Query(..., title="Collection Name", description="Name of the collection"),
    price_lower: float = Query(..., title="Price Lower", description="Lower bound for price"),
    price_upper: float = Query(..., title="Price Upper", description="Upper bound for price")
):
    search_results = search_with_embedding(search_term, collection_name, price_lower, price_upper)
    print("Received search term:", search_term)
    print("Search results:", search_results)

    return {"search_results": search_results}





# search iterator function for complex query 2

def search_iterator(search_term: str, collection_name: str, material: str, color: str):
    try:
        logging.basicConfig(level=logging.INFO)
        embedding = model.encode([search_term])
        print("Generated embedding:", embedding)

        filter_expr = f'{"sold"} == true && {"product_material"} == "{material}" && {"product_color"} == "{color}"'

        client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}", token="root:Milvus")
        print("Milvus client connected:", client)
        collection = Collection(collection_name)

        search_params = {
            "metric_type": "L2",
        }

        iterator = collection.search_iterator(
            expr=filter_expr,
            data=embedding,
            anns_field="name_emb",
            batch_size=100,
            param=search_params,
            output_fields=[
                "product_type", "product_name", "product_description", "sold",
                "product_category", "brand_name", "price_usd",
                "product_material", "product_color"
            ],
            limit=10
        )

        results = []

        while True:
            result = iterator.next()
            if not result:
                iterator.close()
                break

            results.extend(result)

        formatted_results = []
        for hit in results:
            formatted_results.append({
                "product_type": hit.entity.product_type,
                "product_name": hit.entity.product_name,
                "product_description": hit.entity.product_description,
                "sold": hit.entity.sold,
                "product_category": hit.entity.product_category,
                "brand_name": hit.entity.brand_name,
                "price_usd": hit.entity.price_usd,
                "product_material": hit.entity.product_material,
                "product_color": hit.entity.product_color
            })

        return {"search_results": formatted_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying Milvus collection '{collection_name}': {str(e)}")


# complex query 2
# Pronaci 10 najslicnijih rasprodatih proizvoda po imenu. Primeniti filter na osnovu unetih podataka o
# materijalu i boji. 

@app.get("/complex_query_2")
async def perform_search(
    search_term: str = Query(..., title="Search Term", description="Term to search for"),
    collection_name: str = Query(..., title="Collection Name", description="Name of the collection"),
    material: str = Query(..., title="Material", description="Product material"),
    color: str = Query(..., title="Color", description="Product color")
):
    search_results = search_iterator(search_term, collection_name, material, color)
    print("Received search term:", search_term)
    print("Search results:", search_results)

    return {"search_results": search_results}



# search iterator function for complex query 3

def search_iterator_2(search_term: str, collection_name: str):
    try:
        logging.basicConfig(level=logging.INFO)
        embedding = model.encode([search_term])
        print("Generated embedding:", embedding)

        expr_query = f'{"seller_country"} == "Sweden" && {"seller_badge"} == "Trusted" && {"seller_num_followers"} < 500.0'

        client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}", token="root:Milvus")
        print("Milvus client connected:", client)
        collection = Collection(collection_name)

        search_params = {
            "metric_type": "L2",
        }

        iterator = collection.search_iterator(
            expr=expr_query,
            data=embedding,
            anns_field="username_emb",
            batch_size=100,
            param=search_params,
            output_fields = [
                "seller_badge", "seller_username", "seller_country",
                "seller_num_followers"
            ],
            limit=10
        )

        results = []

        while True:
            result = iterator.next()
            if not result:
                iterator.close()
                break

            results.extend(result)

        formatted_results = []
        for hit in results:
            formatted_results.append({
                "seller_badge": hit.entity.seller_badge,
                "seller_username": hit.entity.seller_username,
                "seller_country": hit.entity.seller_country,
                "seller_num_followers": hit.entity.seller_num_followers
            })

        return {"search_results": formatted_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying Milvus collection '{collection_name}': {str(e)}")



# complex query 3
# Pronaci 10 najslicnijih prodavaca po usernamu u kategoriji 'Trusted'. Prodavci su iz Svedske
# i imaju manje od 500 pratioca.


@app.get("/complex_query_3")
async def perform_search(
    search_term: str = Query(..., title="Search Term", description="Term to search for"),
    collection_name: str = Query(..., title="Collection Name", description="Name of the collection")
):
    search_results = search_iterator_2(search_term, collection_name)
    print("Received search term:", search_term)
    print("Search results:", search_results)

    return {"search_results": search_results}



# search with embedding function for complex query 4

def search_with_embedding2(search_term: str, collection_name: str):
    try:

        logging.basicConfig(level=logging.INFO)
        embedding = model.encode([search_term])
        print("Generated embedding:", embedding)

        filter_expr = f'product_season == "All seasons" && product_category == "Women Clothing" && product_name LIKE "Leather%"'

        search_params = {"metric_type": "L2"}

        client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}", token="root:Milvus")
        print("Milvus client connected:", client)

        results = client.search(
            collection_name=collection_name,
            data=embedding,
            anns_field="descr_emb",
            search_params=search_params,
            limit=5,
            filter=filter_expr,
            output_fields = [
                "product_type", "product_name", "product_description", "product_gender_target", "product_season",
                "product_category", "product_condition", "product_like_count", "brand_name", "price_usd"
            ],
        )

        search_results = []

        for hit in results[0]:
            
            product_type = hit['entity']['product_type']
            product_name = hit['entity']['product_name']
            product_description = hit['entity']['product_description']
            product_gender_target = hit['entity']['product_gender_target']
            product_category = hit['entity']['product_category']
            product_season = hit['entity']['product_season']
            product_condition = hit['entity']['product_condition']
            product_like_count = hit['entity']['product_like_count']
            brand_name = hit['entity']['brand_name']
            price_usd = hit['entity']['price_usd']

            search_results.append({
                "product_type": product_type,
                "product_name": product_name,
                "product_description": product_description,
                "product_gender_target": product_gender_target,
                "product_season": product_season,
                "product_category": product_category,
                "product_condition": product_condition,
                "product_like_count": product_like_count,
                "brand_name": brand_name,
                "price_usd": price_usd
            })

            print("Parsed result:", {
                "product_type": product_type,
                "product_name": product_name,
                "product_description": product_description,
                "product_gender_target": product_gender_target,
                "product_season": product_season,
                "product_category": product_category,
                "product_condition": product_condition,
                "product_like_count": product_like_count,
                "brand_name": brand_name,
                "price_usd": price_usd
            })
        
        return search_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying Milvus collection '{collection_name}': {str(e)}")



# complex query 4
# Pronaci 5 najslicnijih proizvoda zenske odece unetom opisu. Proizvodi se mogu nositi u svim sezonama
# i naziv je oblika "Leather%".


@app.get("/complex_query_4")
async def perform_search(
    search_term: str = Query(..., title="Search Term", description="Term to search for"),
    collection_name: str = Query(..., title="Collection Name", description="Name of the collection")
):
    search_results = search_with_embedding2(search_term, collection_name)
    print("Received search term:", search_term)
    print("Search results:", search_results)

    return {"search_results": search_results}




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=INSTANCE_PORT)
