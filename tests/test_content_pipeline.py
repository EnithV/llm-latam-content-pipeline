import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from content_pipeline import LLMContentPipeline


class TestBiasDetection(unittest.TestCase):
    def setUp(self):
        self.pipeline = LLMContentPipeline()

    def test_detects_high_gender_bias(self):
        text = "Only men can lead complex engineering projects."
        result = self.pipeline._detect_bias(text)
        self.assertIn("gender_bias", result)
        self.assertEqual(result["gender_bias"], "high")

    def test_detects_regional_bias_medium(self):
        text = "First world standards should be applied across LATAM."
        result = self.pipeline._detect_bias(text)
        self.assertIn("regional_bias", result)

    def test_clean_technical_text_has_no_high_bias(self):
        text = (
            "The project includes community consultation, regulatory compliance, "
            "and technical analysis for water infrastructure in Colombia."
        )
        result = self.pipeline._detect_bias(text)
        self.assertNotIn("gender_bias", result)


class TestQualityValidation(unittest.TestCase):
    def setUp(self):
        self.pipeline = LLMContentPipeline()

    def _sample_pair(self, response: str, language: str = "es", region: str = "Colombia"):
        return {
            "prompt": "¿Cómo se audita una planta de tratamiento?",
            "response": response,
            "metadata": {
                "category": "water_management",
                "language": language,
                "region": region,
                "cultural_context": "high",
            },
        }

    def test_curated_water_pair_passes_threshold(self):
        pairs = self.pipeline.generate_prompt_response_pairs("water_management", n_pairs=1)
        score, issues = self.pipeline.validate_content_quality(pairs[0])
        self.assertGreaterEqual(score, self.pipeline.quality_threshold)
        self.assertNotIn("Response too short", issues)

    def test_short_response_is_penalized(self):
        pair = self._sample_pair("Muy corto.")
        score, issues = self.pipeline.validate_content_quality(pair)
        self.assertLess(score, self.pipeline.quality_threshold)
        self.assertTrue(any("too short" in issue.lower() for issue in issues))

    def test_regulatory_curated_pair_passes_threshold(self):
        pairs = self.pipeline.generate_prompt_response_pairs("regulatory_frameworks", n_pairs=1)
        score, _ = self.pipeline.validate_content_quality(pairs[0])
        self.assertGreaterEqual(score, self.pipeline.quality_threshold)


class TestLanguageAndCulture(unittest.TestCase):
    def setUp(self):
        self.pipeline = LLMContentPipeline()

    def test_spanish_consistency(self):
        text = "El análisis de cumplimiento normativo en Colombia según la resolución."
        self.assertTrue(self.pipeline._validate_language_consistency(text, "es"))

    def test_cultural_sensitivity_positive_text(self):
        text = (
            "Community participation and inclusive consultation with local stakeholders "
            "and traditional knowledge improve project outcomes."
        )
        score = self.pipeline._assess_cultural_sensitivity(text)
        self.assertGreaterEqual(score, 0.7)


class TestSyntheticGeneration(unittest.TestCase):
    def setUp(self):
        self.pipeline = LLMContentPipeline()

    def test_synthetic_pair_is_substantive(self):
        pair = self.pipeline._generate_synthetic_pair("water_management")
        self.assertGreater(len(pair["response"]), 100)
        self.assertNotIn("Detailed technical response covering", pair["response"])
        score, _ = self.pipeline.validate_content_quality(pair)
        self.assertGreaterEqual(score, 0.65)


class TestEvaluationReport(unittest.TestCase):
    def setUp(self):
        self.pipeline = LLMContentPipeline()

    def test_report_includes_rubric_when_file_exists(self):
        dataset = self.pipeline.generate_training_dataset(size=20)
        report = self.pipeline.create_evaluation_report(dataset)
        self.assertIn("rubric", report)
        self.assertIn("dimensions", report["rubric"])


if __name__ == "__main__":
    unittest.main()
