from vector_db import *
import chromadb
chromadb.PersistentClient('./chromadb').delete_collection('employees')
db = get_db('./chromadb', 'employees')
employee_metadata = [
    {'name': 'ABC', 'age': 20, 'salary': 1000, 'designation': 'SDE'},
    {'name': 'DEF', 'age': 30, 'salary': 2000, 'designation': 'SDE'},
    {'name': 'GHI', 'age': 40, 'salary': 3000, 'designation': 'SDE'},
]

size = 728
employee_embeddings = [
    [1.0]*size,
    [0.0]*size,
    [2.0]*size,
]

add_or_update_employees(employee_embeddings,  employee_metadata=employee_metadata, vector_db=db)
tags = [{'count': 2, 'designation':'SDE'}]
team, team_personality_vector = make_team( vector_db=db, core_member_embedding=[1.0, 1.0, 1.0], member_tags=tags)
print('make_team:', team, team_personality_vector)
print('find_employee:', find_employee('ABC', vector_db=db, n=3))