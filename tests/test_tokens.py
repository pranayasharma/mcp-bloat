import unittest

from mcp_bloat.tokens import count_tokens, counting_method


class TestTokenCounting(unittest.TestCase):
    def test_empty_string_is_zero(self):
        self.assertEqual(count_tokens(""), 0)

    def test_short_text_counts_something(self):
        # "hello" is 1 token under cl100k_base and 2 under chars/4.
        self.assertGreaterEqual(count_tokens("hello"), 1)
        self.assertLessEqual(count_tokens("hello"), 2)

    def test_longer_text_counts_more(self):
        short = count_tokens('{"name": "echo"}')
        long = count_tokens('{"name": "echo", "description": "' + "x" * 400 + '"}')
        self.assertGreater(long, short)

    def test_counting_method_is_a_nonempty_string(self):
        self.assertIsInstance(counting_method(), str)
        self.assertTrue(counting_method())


if __name__ == "__main__":
    unittest.main()
