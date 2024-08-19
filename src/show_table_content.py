import psycopg2

hostname = "localhost"
username = "postgres"
password = "meep"
database = "imagerec"


def fetch_data():
    """function to get and show the written data in the db.
    just for quick checking if it works"""

    try:
        connection = psycopg2.connect(
            host=hostname, user=username, password=password, dbname=database
        )

        cursor = connection.cursor()

        # actual query, changeable to your desire
        # check if there are duplicate names: SELECT name, COUNT(name) AS count FROM images GROUP BY name HAVING COUNT(name) > 1;
        query = """
            SELECT * FROM images_into_db
            ORDER BY id DESC
            LIMIT 10;
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        for row in rows:
            print(row)

    except Exception as error:
        print(f"Fehler Datenaufrufen: {error}")

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


if __name__ == "__main__":
    fetch_data()
