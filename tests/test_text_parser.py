import json
import unittest

from events import extract_event_date_time, to_google_dates
from nlp import *


class TestTextParser(unittest.TestCase):
	def test_date_parser(self):
		json_obj = json.loads('{"text":"event in 1/1/2999, 14:35","timezone":"UTC"}')
		entities = extract_ner(json_obj)
		dates = extract_event_date_time(json_obj["text"], json_obj["timezone"])
		dates = to_google_dates(dates)
		self.assertEqual(len(dates), 2)
		self.assertEqual(dates[0], "29990101T143500Z")
		self.assertEqual(dates[1], "29990101T150500Z")

	def test_two_date_parser(self):
		json_obj = json.loads('{"text":"event end in 1/1/2999 , 14:35 and starts at January 2th at 11:00","timezone":"Asia/Jerusalem"}')
		entities = extract_ner(json_obj)
		dates = extract_event_date_time(json_obj["text"],json_obj["timezone"])
		dates = to_google_dates(dates)
		self.assertEqual(len(dates), 2)
		self.assertEqual(dates[0], "29990101T143500Z")
		self.assertEqual(dates[1], "29990102T110000Z")

if __name__ == '__main__':
	unittest.main()
