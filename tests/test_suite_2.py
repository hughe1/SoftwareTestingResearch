import pytest
from test_source.file1 import *
from test_source.file2 import *


class Tests1:
    def test_1(self):
        result = add_values(1,1)
        assert result == 2
    
    def test_10(self):
        result = arithmetic()
        assert result == "< <= > >= == !="
    
class Tests2:
    def test_2(self):
        result = return3()
        assert result == 3
        
    def test_3(self):
        result = return5()
        assert result == 5

class Tests3:
    def test_4(self):
        result = x_less_than_y(3,5)
        assert result == True
        
    def test_7(self):
        result = five_or_six(3,7)
        assert result == False
