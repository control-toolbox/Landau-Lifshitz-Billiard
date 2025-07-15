import uuid
import hashlib
import os                                 # for saving computations
import json                               # for saving computations

class DataList:
    
    def __init__(self, data_file):
        self.data_file = data_file
   
    def __data_to_code(self, data):
        code=''.encode()
        for key, val in data.items():
            code = code + str(val).encode()
        return code

    def __hash_data(self, data):
        # uuid is used to generate a random number
        salt = uuid.uuid4().hex
        return hashlib.sha256(salt.encode() + self.__data_to_code(data)).hexdigest() + ':' + salt

    def __check_data(self, hashed_data, user_data):
        data, salt = hashed_data.split(':')
        return data == hashlib.sha256(salt.encode() + self.__data_to_code(user_data)).hexdigest()

    def __get_list(self):
        if os.path.isfile(self.data_file):
            file      = open(self.data_file, "r")
            list_data = json.load(file)
            file.close()
        else:
            list_data = None
        return list_data   

    def __data_exists(self, list_data, data):
        v = False
        k = None
        for key in list_data.keys():
            if self.__check_data(key, data):
                v = True
                k = key
        return v, k
    
    def save_list(self, list_data):
        file = open(self.data_file, 'w', encoding='utf-8')
        json.dump(list_data, file)
        file.close()  

    def initiate(self, data, restart):
        # check if the list of data is empty: if yes then create a new list
        # if not empty then check if data has already been created: 
        # if yes then do nothing, else save data
        list_data = self.__get_list()
        if list_data is None:
            list_data = {self.__hash_data(data):data.serialize()}
        else:
            # check if data exists
            v, k = self.__data_exists(list_data, data)
            if v:
                if restart:
                    d = {k:data.serialize()} # update
                    list_data.update(d)
            else:
                d = {self.__hash_data(data):data.serialize()} # new data
                list_data.update(d)
        # save
        self.save_list(list_data)
        #print('Initiate done')
 
    def print(self):
        list_data = self.__get_list()
        if list_data is not None:
            print('List of data:')
            for key, val in list_data.items():
                print('\t' + key)
                print('\t' + str(val))
                print()
        else:
            print('List of data is empty')

    def clear(self):
        list_data = self.__get_list()
        if list_data is not None:
            list_data.clear()
        self.save_list(list_data)
        #print('Clear done')
        
    def update(self, data, new_content):
        list_data = self.__get_list()
        if list_data is None:
            print('You cannot update before initiate')
        else:
            # check if data exists
            v, k = self.__data_exists(list_data, data)
            if v:
                d = list_data[k]
                d.update(new_content)
                list_data.update({k:d})
            else:
                print('You cannot update a data which does not exist')
        # save
        self.save_list(list_data)
        #print('Update done')
        
    def contains(self, data, key):
        ret = False
        list_data = self.__get_list()
        if list_data is not None:
            v, k = self.__data_exists(list_data, data)
            if v:
                d = list_data[k]
                if key in d:
                    ret = True
        return ret
    
    def get(self, data, key):
        ret = None
        list_data = self.__get_list()
        if list_data is not None:
            v, k = self.__data_exists(list_data, data)
            if v:
                d = list_data[k]
                if key in d:
                    ret = d[key]
        return ret

class Error(Exception):
    """
        This exception is the generic class
    """
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class ArgumentTypeError(Error):
    """
        This exception may be raised when one argument of a function has a wrong type
    """

class Data:
    
    def __init__(self, params, data_file, restart=False):
        if(isinstance(params, dict)):
            self.params    = params    # a dict of params
            self.data_file = data_file
            data_list = DataList(self.data_file)
            data_list.initiate(self, restart)
        else:
            raise ArgumentTypeError('params must be a dictionary')

    def items(self):
        return self.params.items()
    
    def serialize(self):
        return self.params
    
    def update(self, new_content):
        if(isinstance(new_content, dict)):
            data_list = DataList(self.data_file)
            data_list.update(self, new_content)
        else:
            raise ArgumentTypeError('new_content must be a dictionary')
        
    def contains(self, key):
        data_list = DataList(self.data_file)
        return data_list.contains(self, key)
    
    def get(self, key):
        data_list = DataList(self.data_file)
        return data_list.get(self, key)
      