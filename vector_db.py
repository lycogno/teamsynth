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

def make_team(core_member_embedding, vector_db, member_tags=[]):
    team = []
    n_members = 1
    team_personality_vector = core_member_embedding
    for member in member_tags:
        new_member = vector_db.query(
            query_embeddings=team_personality_vector,
            where=member,
            n_results=1, 
            include=['metadatas']
        )['metadatas']
        if len(new_member) == 0:
            raise ValueError("No employee found with tag: {}".format(member))
        team.append(new_member[0])
        team_personality_vector *= n_members
        team_personality_vector += new_member[0]
        team_personality_vector /= (n_members + 1)
        n_members += 1
    return team

def get_details(employee):
    return {
        'name': employee['name'],
        'designation': employee['email'],
        'tags': list([(key, value) for key, value in employee if key not in ['name', 'email']])
    }