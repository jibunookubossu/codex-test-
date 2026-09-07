import email.message
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import generate_fuji_videos as app


class FakeResponse:
    def __init__(self, content=b"\x00\x00\x00\x18ftypmp42video"):
        self.content = content
        self.headers = email.message.Message()
        self.headers["Content-Type"] = "video/mp4"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.content


class GeneratorTests(unittest.TestCase):
    def test_all_four_scenes_are_defined(self):
        self.assertEqual(4, len(app.SCENES))
        self.assertTrue(all(name.endswith(".mp4") for name in app.SCENES))

    @patch("urllib.request.urlopen", return_value=FakeResponse())
    def test_generate_requests_and_returns_mp4(self, urlopen):
        result = app.generate("Fuji", "secret", "example/model", 10, 0)
        self.assertIn(b"ftyp", result)
        request = urlopen.call_args.args[0]
        self.assertEqual("Bearer secret", request.headers["Authorization"])
        self.assertNotIn("secret", request.full_url)

    @patch("urllib.request.urlopen", return_value=FakeResponse(b'{"error":"busy"}'))
    def test_generate_rejects_json_even_if_labeled_video(self, _urlopen):
        with self.assertRaisesRegex(app.GenerationError, "unexpected response"):
            app.generate("Fuji", "secret", "example/model", 10, 0)

    def test_mp4_signature_must_be_a_file_type_box(self):
        self.assertFalse(app._is_mp4(b"not-an-ftyp-response"))
        self.assertTrue(app._is_mp4(b"\x00\x00\x00\x18ftypmp42"))

    def test_atomic_save_leaves_no_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.mp4"
            app.save_atomic(path, b"video")
            self.assertEqual(b"video", path.read_bytes())
            self.assertFalse(path.with_suffix(".mp4.part").exists())

    def test_cli_integer_validators(self):
        self.assertEqual(1, app.positive_int("1"))
        self.assertEqual(0, app.nonnegative_int("0"))
        with self.assertRaises(app.argparse.ArgumentTypeError):
            app.positive_int("0")
        with self.assertRaises(app.argparse.ArgumentTypeError):
            app.nonnegative_int("-1")


if __name__ == "__main__":
    unittest.main()
