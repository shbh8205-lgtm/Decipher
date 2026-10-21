import unittest
from io import BytesIO

import zxingcpp
from PIL import Image

from app.barcodes import decode_page_barcodes, verify_central_numbers


def code128_image(value: str = "937309") -> Image.Image:
    barcode = zxingcpp.create_barcode(value, zxingcpp.BarcodeFormat.Code128)
    encoded = zxingcpp.write_barcode_to_image(barcode, scale=3, add_quiet_zones=True)
    height, width = encoded.shape
    generated = Image.frombytes("L", (width, height), memoryview(encoded).cast("B"))
    canvas = Image.new("RGB", (generated.width + 80, generated.height + 80), "white")
    canvas.paste(generated.convert("RGB"), (40, 40))
    return canvas


def image_bytes(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class BarcodeDecodingTests(unittest.TestCase):
    def test_decodes_rotated_code128_with_normalized_bbox(self) -> None:
        rotated = code128_image().rotate(90, expand=True)

        result = decode_page_barcodes(image_bytes(rotated))

        self.assertEqual([item["text"] for item in result], ["937309"])
        self.assertEqual(result[0]["format"], "Code 128")
        self.assertEqual(result[0]["rotation"], 90)
        bbox = result[0]["bbox"]
        self.assertGreater(bbox["width"], 0)
        self.assertGreater(bbox["height"], 0)
        self.assertGreaterEqual(bbox["x"], 0)
        self.assertGreaterEqual(bbox["y"], 0)
        self.assertLessEqual(bbox["x"] + bbox["width"], 1)
        self.assertLessEqual(bbox["y"] + bbox["height"], 1)

    def test_decodes_noisy_code128(self) -> None:
        image = code128_image()
        noise = Image.effect_noise(image.size, 32).convert("RGB")
        noisy = Image.blend(image, noise, 0.08)

        result = decode_page_barcodes(image_bytes(noisy))

        self.assertIn("937309", [item["text"] for item in result])


class CentralNumberVerificationTests(unittest.TestCase):
    @staticmethod
    def barcode(value: str) -> dict[str, str]:
        return {"text": value}

    def test_barcode_ocr_match_status(self) -> None:
        values, result = verify_central_numbers(["937309"], [self.barcode("937309")])
        self.assertEqual(values, ["937309"])
        self.assertEqual(result[0]["status"], "barcode_ocr_match")

    def test_barcode_only_status(self) -> None:
        values, result = verify_central_numbers([], [self.barcode("937309")])
        self.assertEqual(values, ["937309"])
        self.assertEqual(result[0]["status"], "barcode_only")

    def test_ocr_only_status_when_barcode_is_absent(self) -> None:
        values, result = verify_central_numbers(["CENTRAL-1"], [])
        self.assertEqual(values, ["CENTRAL-1"])
        self.assertEqual(result[0]["status"], "ocr_only")

    def test_conflict_uses_barcode_and_preserves_model_alternative(self) -> None:
        values, result = verify_central_numbers(["937308"], [self.barcode("937309")])
        self.assertEqual(values, ["937309"])
        self.assertEqual(result[0]["status"], "conflict")
        self.assertEqual(result[0]["alternatives"], ["937308"])

    def test_multiple_central_values_are_preserved_and_matched_individually(self) -> None:
        values, result = verify_central_numbers(
            ["937309", "937310", "937311"],
            [self.barcode("937309"), self.barcode("937310"), self.barcode("937311")],
        )
        self.assertEqual(values, ["937309", "937310", "937311"])
        self.assertEqual([item["status"] for item in result], ["barcode_ocr_match"] * 3)

    def test_ambiguous_barcodes_only_accept_model_matching_candidates(self) -> None:
        values, result = verify_central_numbers(
            ["937310"],
            [self.barcode("937309"), self.barcode("937310"), self.barcode("937311")],
        )
        self.assertEqual(values, ["937310"])
        self.assertEqual(
            [item["status"] for item in result],
            ["barcode_ocr_match", "barcode_only", "barcode_only"],
        )
        self.assertEqual(result[0]["barcode_value"], "937310")
        self.assertEqual([item["accepted"] for item in result], [True, False, False])

    def test_ambiguous_barcodes_without_ocr_are_not_accepted(self) -> None:
        values, result = verify_central_numbers(
            [],
            [self.barcode("937309"), self.barcode("937310"), self.barcode("937311")],
        )
        self.assertEqual(values, [])
        self.assertEqual([item["accepted"] for item in result], [False, False, False])
        self.assertTrue(all(item["status"] == "barcode_only" for item in result))


if __name__ == "__main__":
    unittest.main()
