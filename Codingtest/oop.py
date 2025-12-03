from pprint import pprint

class Student():
    c=0 # 屬於整個類別的屬性，而不是屬於某個具體的物件實例。所有物件實例共享這個屬性。

    @classmethod
    def hello(cls):
        print(f"hello {cls.c}")

    def __init__(self, name: str) -> None: #實例屬性：與具體物件實例相關聯的屬性。每個物件實例都有自己的屬性值。
        self.__name = name

    def _Say_hello(self, msg: str): #實例方法：定義在類別中的方法，通常用來操作該類別的實例資料。這些方法的第一個參數通常是 self，用來指向當前的物件實例。
        print(f'hello {msg},{self.__name}')
    
    @staticmethod
    def say_hello(): #靜態方法：與類別和物件實例無關的方法。這些方法不會訪問或修改類別屬性或實例屬性。可以使用 @staticmethod 裝飾器來定義靜態方法。
        print("hello")

class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value

    @property
    def area(self):
        return 3.14159 * self._radius ** 2
    


def main():
    a = Student("jack")
    b = Student("Tom")
    a._Say_hello("haha")
    b._Say_hello("123456")
    a._name = 1
    print(a._name)
    pprint(a.__dict__)

    Student.hello()
    print(Student.__name__)

    # 創建一個 Circle 物件
    c = Circle(5)

    # 訪問 radius 屬性
    print(c.radius)  # 輸出: 5

    # 設置 radius 屬性
    c.radius = 10
    print(c.radius)  # 輸出: 10

    # 訪問 area 屬性，注意 area 沒有 setter，因此是只讀的
    print(c.area)  # 輸出: 314.159

    # 嘗試設置一個負值的半徑會引發錯誤
    # c.radius = -3  # 這會拋出 ValueError: Radius cannot be negative

    
if __name__ == '__main__':
    main()