import unittest
from flask_app.app import app

class FlaskAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Creates a test client for the Flask app
        cls.client = app.test_client()

    def test_home_page_loads(self):
        """Test that the home page loads successfully."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Laptop Price Predictor', response.data)

    def test_prediction_post(self):
        """Test POST request to predict route with valid form data."""
        response = self.client.post('/', data={
            'Company': 'Dell',
            'TypeName': 'Notebook',
            'Ram': 8,
            'Weight': 1.98,
            'Touchscreen': 1,
            'Ips': 1,
            'ppi': 141.0,
            'CpuBrand': 'Intel Core i5',
            'HDD': 0,
            'SSD': 256,
            'GpuBrand': 'Intel',
            'os': 'Windows 10'
        })

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'\xe2\x82\xb9', response.data)  # Checks for ₹ symbol in UTF-8

if __name__ == '__main__':
    unittest.main()
