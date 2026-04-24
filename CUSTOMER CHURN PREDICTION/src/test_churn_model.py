import unittest
from pathlib import Path
import pandas as pd
import joblib
from src.train_churn_model import load_data, prepare_features, get_paths

class TestChurnModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_path, cls.models_dir, cls.reports_dir = get_paths()
        cls.model_path = cls.models_dir / "best_churn_model.joblib"

    def test_data_loading(self):
        """Test if data can be loaded and has expected columns."""
        df = load_data(self.data_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("Exited", df.columns)
        self.assertGreater(len(df), 0)

    def test_feature_preparation(self):
        """Test if features are prepared correctly."""
        df = load_data(self.data_path).head(100)
        x, y, cat_cols, num_cols = prepare_features(df)
        
        self.assertEqual(len(x), 100)
        self.assertEqual(len(y), 100)
        self.assertIn("Geography", cat_cols)
        self.assertIn("Gender", cat_cols)
        self.assertNotIn("Exited", x.columns)
        self.assertNotIn("RowNumber", x.columns)

    def test_model_prediction(self):
        """Test if the saved model can make predictions."""
        if not self.model_path.exists():
            self.skipTest("Model file not found. Run training first.")
        
        model = joblib.load(self.model_path)
        df = load_data(self.data_path).head(5)
        x, _, _, _ = prepare_features(df)
        
        predictions = model.predict(x)
        self.assertEqual(len(predictions), 5)
        for pred in predictions:
            self.assertIn(pred, [0, 1])

if __name__ == "__main__":
    unittest.main()
