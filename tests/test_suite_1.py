import pytest
from test_source.file1 import *
from test_source.file2 import *


class Tests1:
    def test_1(self):
        result = add_values(1,1)
        assert result == 2
    
class Tests2:
    def test_2(self):
        result = return3()
        assert result == 3

class Tests3:
    def test_4(self):
        result = x_less_than_y(3,5)
        assert result == True
    
    def test_5(self):
        result = x_more_than_y(3,5)
        assert result == False
        
    def test_6(self):
        result = five_or_six(5,4)
        assert result == True

    def test_8(self):
        result = add_values(100,300)
        assert result == 400
        
    def test_9(self):
        result = logical()
        assert result == "and or not"
