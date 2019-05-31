import re
import os
import subprocess

# 批量调用idaq转asm
path = 'TrainData'
idaPath = 'C:\\Program Files (x86)\\IDA 6.8\\idaq.exe -B '

for i in os.walk(path):
    if i[0] == path:
        continue
    else:
        home, dirs, files = i
        # 遍历每个文件夹
        for file in files:
            if re.search(r'.[o|exe]$', file):  # 可执行文件转asm
                print(os.path.join(home, file) + '  done')
                subprocess.Popen(idaPath + home + os.sep + file)
