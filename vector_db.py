import uuid
import chromadb
def get_db(path, db_name):
    return chromadb.PersistentClient(path).get_or_create_collection(db_name)

def add_or_update_employees(employee_embeddings, employee_metadata, vector_db, employee_ids=[]):
    if not employee_ids:
        employee_ids = [uuid.uuid5(str(employee_embeddings)) for _ in range(len(employee_embeddings))]
    vector_db.upsert(
        embeddings=employee_embeddings,
        metadatas=employee_metadata,
        ids=employee_ids
    )

def make_team(core_member, vector_db, n_members=1, tags=[]):
    return vector_db.query(
        query_embeddings=core_member,
        where=tags,
        n_results=n_members, 
        include=['metadata']
    )