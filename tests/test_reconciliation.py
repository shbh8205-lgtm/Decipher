import unittest

from app.main import reconcile_extractions


def extraction(
    *,
    first_name="Maya",
    central_numbers=None,
    authorization_number="A-100",
    authorization_confidence=0.94,
):
    fields = {
        "first_name": first_name,
        "last_name": "Cohen",
        "id_number": "123456782",
        "central_numbers": central_numbers if central_numbers is not None else ["C-10"],
        "received_date": "12/04/2025",
        "referring_entity": "Clalit",
        "authorizations": [{
            "number": authorization_number,
            "label": "commitment",
            "confidence": authorization_confidence,
            "evidence": "Visible commitment label",
        }],
        "field_confidence": {
            "first_name": 0.9,
            "last_name": 0.9,
            "id_number": 0.9,
            "central_numbers": 0.9,
            "received_date": 0.9,
            "referring_entity": 0.9,
        },
        "evidence": {
            "first_name": "Maya",
            "last_name": "Cohen",
            "id_number": "123456782",
            "central_numbers": "Central sticker",
            "received_date": "Received stamp",
            "referring_entity": "Clalit logo",
        },
        "notes": [],
    }
    return fields


class ReconciliationTests(unittest.TestCase):
    def test_agreement_keeps_values_and_lower_confidence(self) -> None:
        first = extraction(authorization_confidence=0.91)
        second = extraction(authorization_confidence=0.83)

        result = reconcile_extractions(first, second)

        self.assertEqual(result["first_name"], "Maya")
        self.assertEqual(result["field_confidence"]["first_name"], 0.9)
        self.assertEqual(result["authorizations"][0]["confidence"], 0.83)
        self.assertEqual(result["notes"], [])

    def test_disagreement_preserves_candidates_and_flags_review(self) -> None:
        first = extraction(central_numbers=["C-10"])
        second = extraction(first_name="Mia", central_numbers=["C-10", "C-11"], authorization_number="A-101")

        result = reconcile_extractions(first, second)

        self.assertEqual(result["first_name"], "Maya")
        self.assertEqual(result["field_confidence"]["first_name"], 0)
        self.assertEqual(result["central_numbers"], ["C-10", "C-11"])
        self.assertTrue(all(item["confidence"] == 0 for item in result["authorizations"]))
        self.assertTrue(any("Double extraction disagreement" in note for note in result["notes"]))


if __name__ == "__main__":
    unittest.main()
