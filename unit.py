import unittest
import pandas as pd

class TestPreprocessing(unittest.TestCase):
    processed_data = None  # Define processed_data at class scope

    def split_blood_pressure(self, data):
        data[['Systolic', 'Diastolic']] = data['BloodPressure'].str.split('/', expand=True)
        data['Systolic'] = data['Systolic'].astype(float)
        data['Diastolic'] = data['Diastolic'].astype(float)
        data = data.drop(columns=['BloodPressure'])
        return data

    def test_split_blood_pressure(self):
        # Test data
        test_data = pd.DataFrame({
            'BloodPressure': ['120/80', '130/90']
        })
        
        # Expected result
        expected_result = pd.DataFrame({
            'Systolic': [120.0, 130.0],
            'Diastolic': [80.0, 90.0]
        })
        
        # Call preprocessing function and assign to class variable
        TestPreprocessing.processed_data = self.split_blood_pressure(test_data)
        
        # Compare processed data with expected result
        self.assertTrue(TestPreprocessing.processed_data.equals(expected_result))
        
    def tearDown(self):
        print("Test case passed.")
        # Reprint the preprocessed dataset
        print("Preprocessed Dataset:")
        print(TestPreprocessing.processed_data)

if __name__ == '__main__':
    unittest.main()
