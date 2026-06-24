from html.parser import HTMLParser

class HeadingParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_heading = False
        self.current_tag = None
        self.headings = []

    def handle_starttag(self, tag, attrs):
        if tag in ["h1", "h2", "h3", "title"]:
            self.in_heading = True
            self.current_tag = tag

    def handle_endtag(self, tag):
        if tag in ["h1", "h2", "h3", "title"]:
            self.in_heading = False
            self.current_tag = None

    def handle_data(self, data):
        if self.in_heading and data.strip():
            self.headings.append((self.current_tag, data.strip()))

parser = HeadingParser()
with open("docs/Airbnb_Engineering_Blueprint.html", encoding="utf-8") as f:
    parser.feed(f.read())

print("HTML Structure & Headings:")
for tag, text in parser.headings:
    print(f"  {tag.upper()}: {text}")
