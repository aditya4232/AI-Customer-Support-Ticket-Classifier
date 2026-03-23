import unittest
from app import clean_text, predict
import os

class TestCustomerSupportClassifier(unittest.TestCase):
    def test_clean_text(self):
        self.assertEqual(clean_text("My Internet is NOT WoRKiNG!!"), "my internet is not working")

    def test_model_exists(self):
        self.assertTrue(os.path.exists("model.pkl"))

    def test_predict_technical_support(self):
        category, priority = predict("my internet keeps disconnecting")
        self.assertEqual(category, "Technical Support")

    def test_predict_billing(self):
        category, priority = predict("I was charged twice yesterday")
        self.assertEqual(category, "Billing")

    def test_predict_account_management(self):
        category, priority = predict("forgot my password")
        self.assertEqual(category, "Account Management")

if __name__ == '__main__':
    unittest.main()
