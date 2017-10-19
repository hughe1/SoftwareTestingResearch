import pytest
from test_source.file1 import *


class Tests1:
    def test_1(self):
        result = add_values(1,1)
        assert result == 2
    
class Tests2:
    def test_2(self):
        result = return3()
        assert result == 3
        
    def test_3(self):
        result = return5()
        assert result == 5
