import unittest
import models


class ModelsTestCase(unittest.TestCase):
    def test_make_input_col(self):
        self.assertEqual(models.make_input_col("RSI", 10), "RSI_10")
        self.assertEqual(models.make_input_col("RSI_E5", 10), "RSI_10_E5")
        self.assertEqual(models.make_input_col("Williams_R_E5", 10), "Williams_R_10_E5")
        self.assertEqual(models.make_input_col("Close", 10), "Close")
        self.assertEqual(models.make_input_col("MACD", 10), "MACD")
        self.assertEqual(models.make_input_col("MACD_E5", 10), "MACD_E5")
        self.assertEqual(models.make_input_col("STOCH_K", 10), "STOCH_K")
        self.assertEqual(models.make_input_col("STOCH_K_E5", 10), "STOCH_K_E5")


if __name__ == '__main__':
    unittest.main()
