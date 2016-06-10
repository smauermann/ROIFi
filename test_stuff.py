

class Test:
    BLA = 1

    @staticmethod
    def func(bla=BLA):
        print(bla)

t = Test()
t.func(3)
