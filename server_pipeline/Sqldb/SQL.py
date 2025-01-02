import sqlite3
import numpy as np

def create_table():
    conn = sqlite3.connect('face_database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def insert_embedding(name, embedding):
    conn = sqlite3.connect('face_database.db')
    cursor = conn.cursor()
    embedding_blob = embedding.tobytes()
    cursor.execute('INSERT INTO faces (name, embedding) VALUES (?, ?)', (name, embedding_blob))
    conn.commit()
    conn.close()

def retrieve_embeddings(path):
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute('SELECT name, embedding FROM faces')
    data = cursor.fetchall()
    conn.close()

    database = {}
    for name, embedding_blob in data:
        embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        if name not in database:
            database[name] = []
        database[name].append(embedding)
    return database
