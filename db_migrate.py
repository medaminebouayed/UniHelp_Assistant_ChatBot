import psycopg2
conn = psycopg2.connect('postgresql://postgres:password1234@localhost:5433/graphrag')
cur = conn.cursor()
try:
    cur.execute('ALTER TABLE utilisateur ADD COLUMN "estActif" BOOLEAN DEFAULT TRUE;')
    print('estActif column added.')
except Exception as e:
    print('Error adding estActif:', e)

try:
    cur.execute('ALTER TABLE utilisateur ADD COLUMN "dateCreation" TIMESTAMP DEFAULT CURRENT_TIMESTAMP;')
    print('dateCreation column added.')
except Exception as e:
    print('Error adding dateCreation:', e)

conn.commit()
