import unittest
from pricer.data.parser import simplify, scrub, extract_weight, parse
from pricer.data.schema import Item
from pricer.evaluation.metrics import calculate_metrics, calculate_confidence_interval


class TestPricerCore(unittest.TestCase):
    def test_simplify(self):
        text = "  Hello   \n\tWorld!  "
        self.assertEqual(simplify(text), "Hello World!")

    def test_scrub(self):
        scrubbed = scrub("Title", "Part Number: 123", "Features", "Details")
        self.assertNotIn("Part Number", scrubbed)

    def test_extract_weight(self):
        self.assertAlmostEqual(extract_weight("Weight: 10 ounces"), 10.0)
        self.assertAlmostEqual(extract_weight("Package: 2 lbs"), 32.0)
        self.assertEqual(extract_weight("No weight mentioned"), 0.0)

    def test_parse(self):
        dp = {
            "title": "A" * 100,
            "price": "$49.99",
            "features": ["Feature 1" * 20],
            "description": ["Description text" * 20],
            "details": {"Color": "Black", "Weight": "16 oz"},
        }
        item = parse(dp, "Electronics")
        self.assertIsNotNone(item)
        self.assertEqual(item.price, 49.99)
        self.assertEqual(item.category, "Electronics")
        self.assertAlmostEqual(item.weight, 16.0)

    def test_metrics(self):
        truths = [10.0, 20.0, 30.0, 40.0]
        guesses = [12.0, 18.0, 35.0, 40.0]
        metrics = calculate_metrics(truths, guesses)
        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("r2", metrics)
        self.assertGreater(metrics["within_20_pct"], 0)

    def test_confidence_interval(self):
        errors = [5.0, 4.0, 6.0, 5.0, 5.5]
        means, ci, final_mean, final_ci = calculate_confidence_interval(errors)
        self.assertEqual(len(means), len(errors))
        self.assertAlmostEqual(final_mean, 5.1, places=1)


if __name__ == "__main__":
    unittest.main()
