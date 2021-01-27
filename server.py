from socket import *
import os
import shutil
import subprocess

HOST = '2001:da8:270:2020:f816:3eff:fec9:e09b'
PORT = 15660
BUFSIZ = 2048
ADDR = (HOST,PORT)

SerSock = socket(AF_INET6,SOCK_STREAM)
SerSock.bind(ADDR)
SerSock.listen(5)

image_path_base = os.path.dirname(os.path.abspath(__file__))
image_path_local = os.path.join(image_path_base,'SaveResultImage')

print("筛选图片装载路径："+image_path_local)


def remove_last_save_image():
    if os.path.exists(image_path_local):
        shutil.rmtree(image_path_local)
        os.mkdir(image_path_local)
    else:
        os.mkdir(image_path_local)


while True:
    remove_last_save_image()
    print('waiting for connection...')
    conSock, addr = SerSock.accept()
    print("连接客户端地址：",addr)
    data = conSock.recv(BUFSIZ)
    image_require = data.decode('utf-8')
    require_list = image_require.split("-")
    print("目标图片种类: "+require_list[0])
    print("图片筛选速度: "+require_list[1])
    if len(require_list) == 2:
        child = subprocess.Popen('python ./show.py --ImageType '+require_list[0]+' --SpeedSelect '+require_list[1], shell=True)
        child.wait()
    else:
        child = subprocess.Popen('python ./show.py', shell=True)
        child.wait()

    allName_image = sorted(os.listdir(image_path_local))
    conSock.send(str(len(allName_image)).encode('utf-8'))

    for i in range(len(allName_image)):
        name = allName_image[i]
        image_path = os.path.join(image_path_local,name)
        file_size = os.stat(image_path).st_size
        conSock.send(("图片大小:"+str(file_size)).encode('utf-8'))
        statue = conSock.recv(BUFSIZ)
        if statue.decode('utf-8') == "E":
            i = i-1
            continue
        else:
            file = open(image_path, 'rb')
            sent_bytes = 0
            while sent_bytes != file_size:
                file_sent = file.read(BUFSIZ)
                conSock.send(file_sent)
                sent_bytes += len(file_sent)
            file.close()

            print("传输图片 " + name + " 成功！")
            statue = conSock.recv(BUFSIZ)
            if statue.decode('utf-8') == "F":
                i = i-1
    conSock.close()