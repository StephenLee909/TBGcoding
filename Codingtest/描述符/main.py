class RequiredString:

    def __set_name__(self, owner, name):
        self.__property_name = name

    def __set__(self, instance, value):
        if not isinstance(value, str):
            raise Exception(f'{self.__property_name} is not a String')
        
        if len(value)==0:
            raise Exception(f'{self.__property_name} is empty')
        instance.__dict__[self.__property_name] = value
    
    def __get__(self, instance, owner):
        return instance.__dict__[self.__property_name]

class Student2:
    first_name = RequiredString()
    last_name = RequiredString()

def main():
    student2 = Student2()
    student2.first_name = "Jack" #first_name.__set__(student2, "Jack") 
    student2.last_name = "Li"

    student3 = Student2()
    student3.first_name = "Tom"

    print(student2.first_name,student3.first_name)
    print(student2.last_name,student3.last_name)
if __name__ == "__main__":
    main()