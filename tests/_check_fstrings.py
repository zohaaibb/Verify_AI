lines = open("tests/evaluate_200_news.py").read().split("\n")
for i, l in enumerate(lines):
    if 'f"' in l or "f'" in l:
        print("Line {}: {}".format(i+1, l.rstrip()))
