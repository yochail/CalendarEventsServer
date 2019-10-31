from datetime import datetime as dt, timedelta
from dateparser import search

from nlp import *


def create_details(title, text, url):
	return f"""
	{title}
	{text}
	Saved from: {url}
	"""

def extract_type(text, ner_entities):
	return "N/a"

GOOGLE_TIME_FORMATE = "%Y%m%dT%H%M%SZ"


def to_google_dates(dates: List[dt]):
	return [d.strftime(GOOGLE_TIME_FORMATE) for d in dates]


MEETING_DEF_TIME_MIN = 30


def merge_date_time(d: dt, t: dt):
	merged = dt(year=d.year, month=d.month, day=d.day,
	            hour=t.hour, minute=t.minute, second=t.second)
	return merged



def extract_event_date_time(text: str,timezone: str) -> List[str]:
	dates = search.search_dates(text,settings={'TIMEZONE': timezone,"PREFER_DATES_FROM":"future"})

	dates = [d for _,d in dates]
	dates.sort()

	if not dates:
		dates.append(dt.utcnow())

	if len(dates) == 1:
		dates.append(dates[0] + timedelta(minutes=MEETING_DEF_TIME_MIN))

	return to_google_dates([dates[0],dates[1]])

def extract_location(ner_entities:List[Tuple[str,str]]):
	location = [text for text,type in ner_entities
	            if type in [NER_LABELS.GPE,NER_LABELS.LOC,NER_LABELS.ORG,NER_LABELS.FAC]]

	return ' '.join(location)

class event():
	def __init__(self):
		self.title = "New Event"
		self.start_date = dt.now()
		self.end_date = dt.now() + timedelta(minutes=30)
		self.location = ""
		self.details = ""

	def load_event_data(self, json_obj):
		text = json_obj["text"]
		title = json_obj["title"]
		url = json_obj["url"]
		timezone = json_obj["timezone"]
		if title:
			self.title = title
		self.details = create_details(title, text, url)
		ner_entities = extract_ner(json_obj)
		self.start_date, self.end_date = extract_event_date_time(text,timezone)
		self.location = extract_location(ner_entities)
		self.type = extract_type(text, ner_entities)
