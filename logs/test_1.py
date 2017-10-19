F tests/test_1.py::Tests1::()::test_1
 self = <test_1.Tests1 instance at 0x103eb7050>
 
     def test_1(self):
         result = add_values(1,1)
 >       assert result == 2
 E       assert True == 2
 
 tests/test_1.py:8: AssertionError
F tests/test_1.py::Tests2::()::test_2
 self = <test_1.Tests2 instance at 0x103f9e200>
 
     def test_2(self):
         result = return3()
 >       assert result == 3
 E       assert 4 == 3
 
 tests/test_1.py:13: AssertionError
. tests/test_1.py::Tests2::()::test_3
