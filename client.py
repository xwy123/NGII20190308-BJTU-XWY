from socket import *
import os
import shutil


HOST = '2001:da8:270:2020:f816:3eff:fec9:e09b'
PORT = 15660
BUFSIZ = 2048
ADDR = (HOST,PORT)

cliSock = socket(AF_INET6,SOCK_STREAM)
cliSock.connect(ADDR)

image_path_base = os.path.dirname(os.path.abspath(__file__))
save_image_path = image_path_base+str("/receiveImages")

if os.path.exists(save_image_path):
    shutil.rmtree(save_image_path)
    os.mkdir(save_image_path)
else:
    os.mkdir(save_image_path)
print("-----------------------------------------")
print("命令样例:\n" +
      "1.[请输入需求命令]>>>automobile-High\n" +
      "2.[请输入需求命令]>>>automobile-MidHigh\n" +
      "3.[请输入需求命令]>>>automobile-Low")
print("-----------------------------------------\n")
while True:
    msg = input("[请输入需求命令]>>>").strip()
    if len(msg) == 0:
        continue
    else:
        break
cliSock.send(msg.encode())
data = cliSock.recv(BUFSIZ)
image_count = int(data.decode('utf-8'))

for count in range(1, image_count+1):
    file_info = cliSock.recv(BUFSIZ)
    file_name = str(count)+'.jpg'
    file_size_info = []
    try:
        file_size_info = file_info.decode('utf-8').split(':')
        cliSock.send("Y".encode('utf-8'))
    except UnicodeDecodeError:
        print("UnicodeDecodeError...")
        cliSock.send("E".encode('utf-8'))
        count = count-1
        continue

    ImageSave_path = os.path.join(save_image_path,file_name)
    print("图片接收存储路径："+str(ImageSave_path))
    file_size = int(file_size_info[1])

    file = open(ImageSave_path,'ab')
    recive_bytes = 0
    while recive_bytes != file_size:
        image_data = cliSock.recv(BUFSIZ)
        file.write(image_data)
        recive_bytes += len(image_data)
    file.close()
    print("图片 "+file_name+" 收取成功！")
    cliSock.send("F".encode('utf-8'))
cliSock.close()
