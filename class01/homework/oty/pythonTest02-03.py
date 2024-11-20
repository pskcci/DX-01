import numpy as np

num = int(input("input size of matrix: ")) #receive matrix n size
cnt = num*num #multiple n size * n size
list_num = []

for i in range(cnt):
    list_num.append(i+1)

arr_list_num = np.array(list_num)
dim_arr_list_num = np.reshape(arr_list_num,(num,num))

print(dim_arr_list_num)
print(arr_list_num)
