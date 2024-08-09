import psycopg2
import csv

def create_table(conn):
    """Create table in PostgreSQL"""

    #hier können einfach weitere commands hinzugefügt werden, die dann alle ausgeführt werden
    commands = (
        """
        CREATE TABLE IF NOT EXISTS images (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            height INTEGER NOT NULL,
            width INTEGER NOT NULL,
            channels INTEGER NOT NULL,
            avg_blue FLOAT NOT NULL,
            avg_red FLOAT NOT NULL,
            avg_green FLOAT NOT NULL
        )
        """,
    )
    try:
        cur = conn.cursor()
        for command in commands:
            cur.execute(command)
        cur.close()
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)

def import_csv_to_db(conn, csv_path):
    """Import CSV data into PostgreSQL table"""

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header row
        cur = conn.cursor()
        for row in reader:
            cur.execute(
                "INSERT INTO images (id, name, height, width, channels, avg_blue, avg_red, avg_green) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                row
            )
        conn.commit()
        cur.close()

def main():
    conn = None
    try:
        # connect to db
        conn = psycopg2.connect(
            host="localhost",
            database="imagerec",
            user="postgres",
            password="meep"
        )
        
        # create table
        create_table(conn)
        
        # import CSV to DB
        csv_path = "csv/images.csv"
        import_csv_to_db(conn, csv_path)

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

if __name__ == "__main__":
    main()