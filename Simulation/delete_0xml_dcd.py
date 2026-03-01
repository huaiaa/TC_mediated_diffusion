import os
file_list1=os.listdir('./')
# print(file)
# exit(0)
for file1 in file_list1:
    if '0.xml' in file1 or 'data.log' in file1 or '.dcd' in file1:
        os.remove('./{}'.format(file1))


