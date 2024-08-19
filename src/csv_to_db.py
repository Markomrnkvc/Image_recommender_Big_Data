import psycopg2
import csv
from pathlib import Path

csv_path = Path("/Users/mjy/uni_projects/Image_recommender_Big_Data/src/csv/images.csv")


def create_table(conn):
    """Create table in PostgreSQL"""

    # hier können einfach weitere commands hinzugefügt werden, die dann alle ausgeführt werden
    commands = (
        """
        CREATE TABLE IF NOT EXISTS images_into_db (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL
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

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header row
        cur = conn.cursor()
        for row in reader:
            # only take the first 2 columns (id, name)
            row_to_insert = row[:2]
            print(row_to_insert)

            # double check if there are only 2 values
            if len(row_to_insert) == 2:
                cur.execute(
                    "INSERT INTO images_into_db (id, name) VALUES (%s, %s)",
                    row_to_insert,
                )
            else:
                print(f"Unexpected row length: {len(row_to_insert)} elements")

        # for row in reader:
        # cur.execute(
        # "INSERT INTO images (id, name) VALUES (%s, %s)",
        # row
        # )
        conn.commit()
        cur.close()


def main():
    conn = None
    try:
        # connect to db
        conn = psycopg2.connect(
            host="localhost", database="imagerec", user="postgres", password="meep"
        )

        # create table
        create_table(conn)

        # import CSV to DB
        import_csv_to_db(conn, csv_path)

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    main()
