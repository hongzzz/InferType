typeInfo = {
    'signed': 0,
    'unsigned': 1,
    'both not': 2
}

signedList = ['int', 'long long int', '']

unsignedList = ['unsigned int', 'long long unsigned int', '']

def getType(typeStr):
    if typeStr in signedList:
        return 'signed'
    elif typeStr in unsignedList:
        return 'unsigned'
    else:
        return 'both not'


print(getType('unsigned int'))