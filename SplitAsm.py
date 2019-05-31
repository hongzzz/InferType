import re
import os
import shutil

intputPath = 'asmFile/'
outputPath = 'TestData/'

def SplitIDAasm(pathFile, outputPath):
    if outputPath.endswith('/'):
        pass
    else:
        outputPath = outputPath + '/'

    appName = pathFile.split("/")[-1]
    appName = appName.replace('.asm', '')

    with open(pathFile, 'r', encoding='utf-8') as file:
        fullContext = file.readlines()
        linenum = len(fullContext)
        functionProtoytpe = set()
        head_end = []

        for i in range(0, linenum):

            matText = re.search(r'^(\.|HEADER).+', fullContext[i])
            if matText:
                if '  ' in fullContext[i]:
                    a, b = fullContext[i].split('  ', 1)
                    fullContext[i] = b
                else:
                    fullContext[i] = ''
            else:
                pass

            if '== S U B	R O U T	I N E ==' in fullContext[i] or '== S U B R O U T I N E ==' in fullContext[i]:
                # 有subroutine的是开头
                head = i
                end = 0
                functionName = ''
                for j in range(i + 1, linenum):
                    # 看看有没有proc near
                    if 'proc near' in fullContext[j]:
                        if '\t' not in fullContext[j]:
                            fullContext[j] = fullContext[j].strip()
                            fullContext[j] = fullContext[j].replace(' ', '\t', 1)
                        functionName, rest = fullContext[j].split('\t', 1)
                    # 找结束的地方
                    if '===============' in fullContext[j]:
                        if functionName == '':  # 如果这个函数中没有proc near标识，那就不管它了
                            break
                        else:
                            end = j
                            break
                if functionName != '' and end != 0:
                    # functionName = re.sub('[^A-Za-z]', '', functionName)
                    head_end.append({'functionName': functionName, 'head': head, 'end': end})

            # 还有一些附加的函数声明
            if re.search(r';.*\(.*\)', fullContext[i]) and 'extrn' in fullContext[i + 1]:
                functionProtoytpe.add(fullContext[i])

        # 该文件已经存在则先删除 再新建
        try:
            if os.path.exists(outputPath + appName):
                shutil.rmtree(outputPath + appName)
            if not os.path.exists(outputPath + appName):
                os.makedirs(outputPath + appName)
        except:
            pass

        # 输出分割后的文件
        for each in head_end:
            with open(outputPath + appName + '/' + str(each['functionName']) + '().txt', 'w',
                      encoding='utf-8') as newfile:
                for i in range(each['head'] + 4, each['end']):
                    newfile.write(fullContext[i])

        if len(functionProtoytpe) != 0:
            with open(outputPath + appName + '/' + 'function_prototype', 'w', encoding='utf-8') as fpfile:
                for each in functionProtoytpe:
                    fpfile.write(each)


for i in os.walk(intputPath):
    if i[0] != intputPath:
        continue
    else:
        for home, dirs, files in os.walk(i[0]):
            for each in files:
                if each.endswith('.asm'):
                    pathFile = home + each
                    SplitIDAasm(pathFile, outputPath)
                    print(each)

print('(All splited)')
