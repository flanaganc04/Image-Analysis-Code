import csv

BigData = []
for x in range(10):
    for n in range(5):
        data = {}
        data['sample'] = f'{x+1}'
        data['author'] = 'Colin'
        data['sub'] = f'{n+1}'
        BigData.append(data)
        print(n+1)
        
file_name = 'csvPractice/output3.csv'
with open(file_name, mode = 'w', newline = '') as file:
    writer = csv.DictWriter(file, fieldnames = BigData[0].keys())
    writer.writeheader()
    writer.writerows(BigData)

# # Open a file for reading ('r' mode)
# with open('file.txt', 'r') as file:
#     content = file.read()
#     print(content)

# # Open a file for writing ('w' mode)
# with open('new_file.txt', 'w') as file:
#     file.write('Hello, World!')

# # Open a file for appending ('a' mode)
# with open('log.txt', 'a') as file:
#     file.write('New log entry\n')

# # Open a file for both reading and writing ('r+' mode)
# with open('data.txt', 'r+') as file:
#     content = file.read()
#     file.write('Appending new data')

# # Open a file for both writing and reading ('w+' mode)
# with open('file.txt', 'w+') as file:
#     file.write('Writing and reading\n')
#     file.seek(0)  # Reset file pointer to beginning
#     content = file.read()
#     print(content)