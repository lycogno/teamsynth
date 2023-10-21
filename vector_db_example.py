from vector_db import *
import chromadb
chromadb.PersistentClient('./chromadb').delete_collection('employees')
db = get_db('./chromadb', 'employees')
employee_metadata = [
    {'name': 'ABC', 'age': 20, 'salary': 1000, 'designation': 'SDE'},
    {'name': 'DEF', 'age': 30, 'salary': 2000, 'designation': 'SDE'},
    {'name': 'GHI', 'age': 40, 'salary': 3000, 'designation': 'SDE'},
]

add_or_update_employees([[1, 1, 1], [0, 0, 1], [1, 2, 0]], employee_metadata=employee_metadata, vector_db=db)
tags = [{'count': 2, 'designation':'SDE'}]
team, team_personality_vector = make_team([1.0, 1.0, 1.0], vector_db=db, member_tags=tags)
print(team, team_personality_vector)