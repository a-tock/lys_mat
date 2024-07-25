import unittest
from unittest.mock import patch
import numpy as np
from lys_mat import Atom

class AtomTests(unittest.TestCase):

    def setUp(self):
        self.atom = Atom('H', (0, 0, 0))

    def test_duplicate_returns_deep_copy(self):
        # Test that the duplicate method returns a deep copy of the object.
        duplicate_atom = self.atom.duplicate()
        self.assertIsNot(duplicate_atom, self.atom)
        self.assertEqual(duplicate_atom.Element, self.atom.Element)
        self.assertEqual(duplicate_atom.Position.tolist(), self.atom.Position.tolist())

    def test_duplicate_does_not_modify_original_object(self):
        # Test that the duplicate method does not modify the original object.
        original_position = self.atom.Position.copy()
        duplicate_atom = self.atom.duplicate()
        duplicate_atom.Position[0] += 1
        self.assertEqual(self.atom.Position.tolist(), original_position.tolist())

    def test_duplicate_copies_attributes(self):
        # Test that the duplicate method copies all attributes of the object.
        self.atom.Occupancy = 0.5
        self.atom.Uani = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        duplicate_atom = self.atom.duplicate()
        self.assertEqual(duplicate_atom.Occupancy, self.atom.Occupancy)
        self.assertTrue(np.array_equal(duplicate_atom.Uani, self.atom.Uani))

    def test_duplicate_copy_methods(self):
        # Test that the duplicate method does not copy methods of the object.
        duplicate_atom = self.atom.duplicate()
        self.assertTrue(hasattr(duplicate_atom, 'duplicate'))

    @patch('copy.deepcopy')
    def test_duplicate_calls_deepcopy(self, mock_deepcopy):
        # Test that the duplicate method calls the deepcopy function.
        self.atom.duplicate()
        mock_deepcopy.assert_called_once_with(self.atom)