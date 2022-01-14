# -*- coding: utf-8 -*-
import unittest
from Feature import Feature_MS2 as ms2
from test_data import parsedMGF as mgfData

#Note about test mgfData
# -This file has 4 precursor ions with the following figures of merit
#   feature 1 - 111.0915613 m/z, 2.7383 min, 1+ charge
#   feature 2 - 111.0915613 m/z, 3.0269 min, 1+ charge,
#   feature 3 - 111.0915613 m/z, 3.09486666666667 min, 0 charge
#   feature 4 - 111.09151613 m/z, 5.0496 min, 0 charge

"""Current implementation is WIP and need expansion/further consideration"""

test1 = ms2.Feature_List_Constructor(mgfData)
test2 = ms2.Feature_List_Constructor(mgfData, mass_accuracy = 0, rt_accuracy = 0)
test3 = ms2.Feature_List_Constructor(mgfData, mass_accuracy = 0, rt_accuracy = 10)
test4 = ms2.Feature_List_Constructor(mgfData, mass_accuracy = 10, rt_accuracy = 0)
test5 = ms2.Feature_List_Constructor(mgfData, mass_accuracy = 100, rt_accuracy = 100)

class TestFeature_List_Constructor(unittest.TestCase):
    def test_generate_feature_list(self):
        self.assertEqual(len(test1.Feature_List), 3)
        self.assertEqual(len(test2.Feature_List), 4)
        self.assertEqual(len(test3.Feature_List), 4)
        self.assertEqual(len(test4.Feature_List), 4)
        self.assertEqual(len(test5.Feature_List), 1)

class TestMS2_Feature(unittest.TestCase):
    def test_generate_feature_list(self):
        pass
    
class TestMS2_Spectrum(unittest.TestCase):
    def test_generate_feature_list(self):
        pass
    
if __name__ == '__main__':
    unittest.main()
