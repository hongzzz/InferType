import os
import json
import pickle
import re
import copy


def getList(path):
    with open(path, encoding='utf-8') as f:
        tCodeList = []
        tVarList = []
        for line in f.readlines():
            line = line.split(';', 1)[0]  # 去除注释
            line = line.strip()
            if line:
                if '=' in line:  # 有=符号可能是变量
                    line = line.split('=', 1)[0]
                    line = line.strip()
                    tVarList.append(line)
                else:
                    if '\t' in line:  # tab或者4个空格
                        arr = line.split('\t', 1)
                    else:
                        arr = line.split(' ', 1)

                    if len(arr) >= 2:
                        arr = [arr[0]] + arr[1].split(',')
                    for i in range(len(arr)):
                        arr[i] = arr[i].strip()
                    # print(arr)
                    tCodeList.append(arr)
    return tVarList, tCodeList


def getVarType(s):
    if re.search(r'\[.+\]', s):
        res = '_'
    elif re.search(r'xmm[0-7]', s):
        res = 'xmm'
    elif re.search(r'(?:eax|ebx|ecx|edx|esi|edi)(?!\w)', s):
        res = 'reg'
    elif re.search(r'(?:rax|rbx|rcx|rdx|rsi|rdi)(?!\w)', s):
        res = 'rregg'
    elif re.search(r'(?:ax|bx|cx|dx|si|di)(?!\w)', s):
        res = 'ax'
    elif re.search(r'(?:al|bl|cl|dl|ah|bh|ch|dh)(?!\w)', s):
        res = 'al'
    else:
        res = 'imm'
    return res


def getRealType(path):
    realType = []
    try:
        # 打开从源代码提取的类型信息的文件
        with open(path, encoding='utf-8') as typefile:
            tempDict = {'functionName': '', 'varType': []}
            for line in typefile.readlines():
                if "function:" in line:
                    left, right = line.strip().split(':', 1)
                    left, right = right.strip().split('(', 1)
                    tempDict['functionName'] = left.strip()
                elif "parameter:" in line:
                    left, right = line.strip().split(':', 1)
                    p = re.compile('[,|)]')
                    right = p.sub("", right)
                    p = re.compile('\[.*\]')
                    right = p.sub("[]", right)
                    if '=' in right:
                        right = right.split('=')[0]
                    right = right.strip().strip(';')
                    right = right.rsplit(' ', 1)
                    tempDict['varType'].append(right)
                elif "variable:" in line:
                    left, right = line.strip().split(':', 1)
                    declar = right
                    p = re.compile('\[.*\]')  # 去除一些无用的符号
                    declar = p.sub("[]", declar)
                    if '=' in declar:
                        declar = declar.split('=')[0]
                    declar = declar.strip().strip(';')
                    declar = declar.rsplit(' ', 1)
                    tempDict['varType'].append(declar)
                elif "----------" in line:
                    if not tempDict['functionName'] == '':
                        realType.append(tempDict)
                        # print(tempDict)
                    tempDict = {'functionName': '', 'varType': []}
                else:
                    pass
                # print(realType)
            return realType
    except IOError as err:
        print('File error:' + str(err))


def getData(varList, codeList):
    matrixList = []
    for var in varList:
        # print('------' + t + '------')
        index = 0
        matrix = [0] * len(InstructionList)
        regSet = set()
        regSet.add(var)
        tempCodeList = copy.deepcopy(codeList)

        while len(regSet) != 0:
            item = regSet.pop()
            for c in tempCodeList:
                flag1 = None
                flag2 = None
                if len(c) <= 1:  # 无参数直接过滤
                    index += 1
                    continue

                flag1 = re.search(r'[^\w]' + item + r'[^\w]', c[1])
                if len(c) == 3:
                    flag2 = re.search(r'[^\w]' + item + r'[^\w]', c[2])

                if flag1 or flag2:
                    if flag1:
                        c[1] = getVarType(c[1])
                        if len(c) > 2:
                            temp = getVarType(c[2])
                            if temp != 'imm':
                                regSet.add(c[2])
                            c[2] = temp
                    elif flag2:
                        c[2] = getVarType(c[2])
                        temp = getVarType(c[1])
                        if temp != 'imm':
                            regSet.add(c[1])
                        c[1] = temp
                    s = ' '.join(c)

                    try:
                        n = InstructionList.index(s)
                        matrix[n] += 1
                        # print(matrix)
                    except:
                        print(s + ' ----------------------------------lose')
                index += 1

        matrixList.append(matrix)

    return matrixList  # 返回向量列表（下标对应）


def type2Label(str):
    if 'unsigned' in str:
        return 0  # 'unsigned'
    elif 'void' in str or 'float' in str or 'double' in str:
        return 2  # 'neither'
    else:
        return 1  # 'signed'


with open('common/InstructionList.json', 'rb') as f:
    InstructionList = json.load(f)
    # print(InstructionList)

allDataList = []
allLabelList = []

dirPath = 'TrainData'

for i in os.walk(dirPath):
    if i[0] == dirPath:
        continue
    else:
        home, dirs, files = i
        dirName = home.split('\\')[1]
        realTypeList = getRealType(os.path.join(home, dirName + '.txt'))
        # print(realTypeList)
        if len(realTypeList) == 0:
            print('can not find type defined file')
            continue

        for file in files:
            if not file.endswith('.txt'):  # only txt
                continue

            if file == dirName + '.txt':  # 变量类型声明文件忽略
                continue

            flag = re.search(r'\b(\w*)\(', file)

            if not flag:
                continue

            wholePath = os.path.join(home, file)
            functionName = flag.group(1)
            # print(wholePath)
            varList, codeList = getList(wholePath)
            matrixList = getData(varList, codeList)

            for obj in realTypeList:
                if obj['functionName'] == functionName:
                    types = obj['varType']
                    # print(home + '    ' + functionName + '----------')
                    for i in range(len(varList)):
                        for x in types:
                            t, v = x  # type,varName
                            # print(v, varList[i], re.search(r'\b' + varList[i] + r'\b', v))
                            r = re.search(r'\b' + varList[i] + r'\b', v)
                            if r:  # 存在对应变量才存入样本
                                # print(varList[i])
                                allDataList.append(matrixList[i])
                                if r.start() > 0 and v[r.start() - 1] == '*':
                                    allLabelList.append(t + '*')
                                elif r.end() < len(v) and v[r.end()] == '[':
                                    allLabelList.append(t + '*')
                                else:
                                    allLabelList.append(t)
                    break

# 将类型转化为signed, unsigned, neither
for i in range(len(allLabelList)):
    # print(allLabelList[i])
    # print(type2Label(allLabelList[i]))
    allLabelList[i] = type2Label(allLabelList[i])

print(len(allDataList))
print(len(allLabelList))

with open('common/data.pkl', 'wb') as file:
    pickle.dump(allDataList, file)
with open('common/label.pkl', 'wb') as file:
    pickle.dump(allLabelList, file)
