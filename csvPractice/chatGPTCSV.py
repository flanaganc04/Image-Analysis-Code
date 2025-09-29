import csv

# Sample data
data = [
    {'Name': 'Alice', 'Age': 28, 'City': 'New York'},
    {'Name': 'Bob', 'Age': 34, 'City': 'Los Angeles'},
    {'Name': 'Charlie', 'Age': 22, 'City': 'Chicago'}
]

# Specify the file name
file_name = 'csvPractice/output.csv'

# Writing to csv file
with open(file_name, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)

print(f"Data has been written to {file_name}")