import pickle

def savePickle(filename,data):
    with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def loadPickle(filename):
    with open(filename, 'rb') as handle:
            data = pickle.load(handle)
    return data