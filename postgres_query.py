import psycopg2

try:
    connection = psycopg2.connect(user = "postgres",
                                  password = "1qaz!QAZ",
                                  host = "114.203.211.52",
                                  port = "49153",
                                  database = "anhdb")

    cursor = connection.cursor()
    # Print PostgreSQL Connection properties
    print ( connection.get_dsn_parameters(),"\n")

    # Print PostgreSQL version
    #cursor.execute("SELECT version();")
    cursor.execute("SELECT distinct name FROM projects_with_repository_fields WHERE name like '%1vbvdk84ol%'")
    record = cursor.fetchone()
    #print("You are connected to - ", record,"\n")
    print(record)

except (Exception, psycopg2.Error) as error :
    print ("Error while connecting to PostgreSQL", error)
finally:
    #closing database connection.
        if(connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")