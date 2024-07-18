import os
import random
import time
file_name = "motto.txt"
path = os.path.dirname(os.path.abspath(__file__))

# Open the file for read
#1
f_handle = open(path+'/'+file_name,"a")
f_handle.write('Fiat Lux')
#2
f_handle.close()
f_handle = open(path+'/'+file_name,"r")
print(f_handle.read())
#3
f_handle.close()
f_handle = open(path+'/'+file_name,"a")
f_handle.write('\nlet there be light')