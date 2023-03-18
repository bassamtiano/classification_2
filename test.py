
class TestManggil():
    def __init__(self, nama) -> None:
        self.nama = nama
    
    def forward(self, test_param):
        print(self.nama)
        print(test_param)
        
    def __call__(self, test_param):
        self.forward(test_param)
        
    
        
if __name__ =="__main__":
    tm = TestManggil(nama = "nibras")
    tm(test_param = "hujan")
