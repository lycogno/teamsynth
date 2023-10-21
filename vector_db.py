import uuid
import chromadb

def get_db(path, db_name):
    return chromadb.PersistentClient(path).get_or_create_collection(db_name)

def add_or_update_employees(employee_embeddings, employee_metadata, vector_db, employee_ids=[]):
    if not employee_ids:
        employee_ids = [str(uuid.uuid4()) for _ in range(len(employee_embeddings))]
    vector_db.upsert(
        embeddings=employee_embeddings,
        metadatas=employee_metadata,
        ids=employee_ids
    )

def make_team(core_member_embedding, vector_db, member_tags=[]):
    team = []
    n_members = 0
    team_personality_vector = core_member_embedding
    for member in member_tags:
        count = 1
        if 'count' in member.keys():
            count = member['count']
            member.pop('count')
        new_member = vector_db.query(
            query_embeddings=team_personality_vector,
            where=member,
            n_results=count, 
            include=['metadatas', 'embeddings']
        )
        if len(new_member) == 0:
            raise ValueError("No employee found with tag: {}".format(member))
        for j in range(count):
            team.append(new_member['metadatas'][0][j])
            team_personality_vector = [i*n_members for i in team_personality_vector]
            team_personality_vector = [team_personality_vector[i] + new_member['embeddings'][0][j][i] for i in range(len(team_personality_vector))]
            team_personality_vector = [i/(n_members + 1) for i in team_personality_vector]
            n_members += 1
    return team, team_personality_vector

def find_employee(name, vector_db, n):
    a = vector_db.query(
        query_embeddings=[0]*n,
        where={'name': name},
        n_results=1,
        include=['embeddings', 'metadatas']
    )
    return a['embeddings'][0][0], a['metadatas'][0][0]