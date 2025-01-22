import pandas as pd
from sqlalchemy import create_engine

# Database connection details
db_user = 'root'  # Replace with your MySQL username
db_password = 'b5CaQ9WK2'  # Replace with your MySQL password
db_host = '127.0.0.1'  # Replace with your MySQL host
db_name = 'chess'  # Replace with your database name
table_name = 'games'  # Table name to insert data into

# Filepath to the CSV
csv_file_path = './output.csv'

# Create the database connection using SQLAlchemy
engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")

# Read the CSV file into a pandas DataFrame
print("Reading the CSV file...")
df = pd.read_csv(csv_file_path)

# Write the DataFrame to the MySQL database
print("Writing data to the database...")
df.to_sql(table_name, engine, if_exists='append', index=False)

print("Data successfully written to the database.")
