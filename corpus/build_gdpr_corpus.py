"""
Build the GDPR knowledge base from the authoritative EUR-Lex text.

    python corpus/build_gdpr_corpus.py            # fetch + parse + write JSONL
    python corpus/build_gdpr_corpus.py --check    # re-parse the cached HTML only

Source: Regulation (EU) 2016/679, CELEX 32016R0679, from EUR-Lex — the Official
Journal text. Not gdpr-info.eu or any other mirror: in a system whose whole claim is
that its citations can be checked, the corpus has to come from the authority the
citation names, or the citation is decorative.

Why this produces JSONL and not a PDF
-------------------------------------
The rest of this repo ingests PDFs and cites "file, page 14". For a regulation that
is the wrong unit: nobody looks up page 14 of the GDPR, they look up Article 17(1)(a),
and the page a provision lands on is an artefact of the typesetting. Rendering the
regulation to a PDF and parsing it back would destroy exactly the structure that makes
a legal citation checkable. So the structure is extracted once, here, and carried into
the vector store as metadata.

Chunk unit: the article paragraph
---------------------------------
One record per numbered paragraph (Article 17(1)), with its sub-points (a), (b), (c)
inlined rather than split out. A point on its own is not a statement of law: "(a) the
personal data are no longer necessary..." only means anything under the chapeau that
governs it ("the data subject shall have the right to obtain erasure where one of the
following grounds applies"). Retrieving the point without its chapeau returns a
fragment that reads like an answer and is not one.
"""
import argparse
import html
import json
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CACHED_HTML = os.path.join(HERE, ".gdpr_en.html")
OUT_JSONL = os.path.join(HERE, "gdpr_en.jsonl")

CELEX = "32016R0679"
SOURCE_URL = f"https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{CELEX}"
SHORT_NAME = "GDPR"
LONG_NAME = "Regulation (EU) 2016/679 (General Data Protection Regulation)"

# The enacting terms start at the div marked enc_1; everything before it is citations
# and the 173 recitals. Recitals are interpretive aids, not binding provisions, and
# citing one as though it were an obligation is a domain error, so they are excluded.
ENACTING_TERMS_MARKER = 'id="enc_1"'

# The enacting terms end at the final formula ("This Regulation shall be binding in
# its entirety..."), after which come the signatures and the 21 footnotes. The last
# article has no following article to bound it, so without this the closing provision
# absorbs all of that — and then a question about accreditation retrieves Article 99
# and cites the entry-into-force rule for something it does not say. A citation that
# points at the wrong provision is worse than no answer, which is the whole reason
# this corpus is built structurally in the first place.
END_OF_ENACTING_RE = re.compile(r'class="oj-(?:final|signatory)"')

ARTICLE_DIV_RE = re.compile(r'<div class="eli-subdivision" id="art_(\d+)">')
ARTICLE_NUM_RE = re.compile(r'class="oj-ti-art">\s*Article\s+(\d+)\s*<', re.I)
ARTICLE_TITLE_RE = re.compile(r'class="oj-sti-art">(.*?)</p>', re.S)
CHAPTER_RE = re.compile(r'>\s*CHAPTER\s+([IVXLC]+)\s*<', re.I)
CHAPTER_TITLE_RE = re.compile(r'id="cpt_[IVXLC]+\.tit_1">.*?class="oj-ti-section-2">(.*?)</p>', re.S)
# Paragraph containers inside an article: <div id="017.001">
PARA_DIV_RE = re.compile(r'<div id="(\d{3}\.\d{3})">')
# A sub-point renders as a one-row, two-cell table: label cell "(a)", then its text.
# Three definitions in Article 4 — (16), (22) and (23) — nest a further table inside
# their text cell, so the cell boundary cannot be found by scanning to the next
# "</td>": that lands on the closing tag of the *inner* label cell and silently
# truncates the definition. Table spans are matched by counting depth instead.
TABLE_BOUNDARY_RE = re.compile(r"<table\b|</table>", re.I)
LABEL_CELL_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.S | re.I)
LABEL_RE = re.compile(r"^\(([a-z0-9]+)\)$", re.I)
# A leading "1.   " on a paragraph repeats the number we already hold as metadata.
LEADING_NUM_RE = re.compile(r'^\d+\.\s+')
TAG_RE = re.compile(r'<[^>]+>')


def strip_tags(fragment: str) -> str:
    """Plain text from an HTML fragment, with entities and whitespace normalised."""
    # Slicing at the final formula can leave a half-written tag at the end of the
    # fragment, which TAG_RE will not match because it never sees the closing ">".
    text = re.sub(r"<[^>]*$", " ", fragment)
    text = TAG_RE.sub(" ", text)
    text = html.unescape(text)
    # EUR-Lex uses non-breaking spaces liberally, including inside "Article 17(1)".
    text = text.replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def fetch_html() -> str:
    """Download the Official Journal HTML, caching it so re-parsing is offline."""
    req = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    with open(CACHED_HTML, "w", encoding="utf-8") as fh:
        fh.write(raw)
    return raw


def load_html(use_cache_only: bool) -> str:
    if use_cache_only:
        if not os.path.exists(CACHED_HTML):
            sys.exit(f"No cached HTML at {CACHED_HTML}. Run without --check first.")
        with open(CACHED_HTML, encoding="utf-8") as fh:
            return fh.read()
    return fetch_html()


def split_articles(enacting: str):
    """Yield (article_number, article_html) for each article, in document order."""
    matches = list(ARTICLE_DIV_RE.finditer(enacting))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(enacting)
        yield int(match.group(1)), enacting[match.start():end]


def chapter_index(enacting: str):
    """Map each character offset to the chapter in force at that point."""
    marks = []
    for match in CHAPTER_RE.finditer(enacting):
        title_match = CHAPTER_TITLE_RE.search(enacting, match.end(), match.end() + 2000)
        title = strip_tags(title_match.group(1)) if title_match else ""
        marks.append((match.start(), match.group(1).upper(), title))
    return marks


def chapter_at(marks, offset: int):
    current = ("", "")
    for start, numeral, title in marks:
        if start <= offset:
            current = (numeral, title)
        else:
            break
    return current


def top_level_tables(fragment: str):
    """The spans of tables in this fragment that are not nested inside another."""
    spans, depth, start = [], 0, None
    for match in TABLE_BOUNDARY_RE.finditer(fragment):
        if match.group(0).lower().startswith("<table"):
            if depth == 0:
                start = match.start()
            depth += 1
        elif depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                spans.append(fragment[start:match.end()])
                start = None
    return spans


def parse_points(fragment: str):
    """
    Extract (label, text) for each sub-point, folding nested points into their parent.

    Article 4(23) reads "'cross-border processing' means either: (a) ... or (b) ...".
    The (a)/(b) limbs are alternatives *within* the definition, not definitions in
    their own right, so they belong in its text rather than beside it.
    """
    points = []
    for table in top_level_tables(fragment):
        # The label cell never contains a nested table, so the first </td> closes it.
        label_match = LABEL_CELL_RE.search(table)
        if not label_match:
            continue
        label_match_text = LABEL_RE.match(strip_tags(label_match.group(1)))
        if not label_match_text:
            continue

        rest = table[label_match.end():]
        nested = top_level_tables(rest)
        own = rest
        for span in nested:
            own = own.replace(span, " ")
        text = strip_tags(own)
        for sub_label, sub_text in parse_points(rest):
            text = f"{text} ({sub_label}) {sub_text}".strip()
        points.append((label_match_text.group(1), text))
    return points


def chapeau_of(fragment: str) -> str:
    """
    The paragraph's own text, excluding its sub-point tables.

    Everything before the first <table> is the chapeau; when there is no table the
    whole fragment is the text.
    """
    head = fragment.split("<table", 1)[0]
    return LEADING_NUM_RE.sub("", strip_tags(head))


def parse_paragraphs(article_html: str):
    """
    Yield (paragraph_number_or_None, text, points) for one article.

    Articles whose text is not split into numbered paragraphs (a single block) yield
    one entry with paragraph None, so a citation to them reads "Article 21" rather
    than inventing a paragraph that does not exist.
    """
    matches = list(PARA_DIV_RE.finditer(article_html))
    if not matches:
        body = article_html.split("</p>", 2)[-1]
        text = chapeau_of(body)
        points = parse_points(body)
        if text or points:
            yield None, text, points
        return

    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(article_html)
        fragment = article_html[match.start():end]
        # id "017.001" -> paragraph 1
        paragraph = int(match.group(1).split(".")[1])
        yield paragraph, chapeau_of(fragment), parse_points(fragment)


def citation_for(article: int, paragraph) -> str:
    return f"{SHORT_NAME} Art. {article}" + (f"({paragraph})" if paragraph else "")


def is_definition_list(paragraph, points) -> bool:
    """
    Whether an article is a list of independently citable numbered points.

    Article 4 is the only one in the GDPR: an article with no numbered paragraphs
    whose content is numbered points. Its definitions are cited individually —
    'controller' is Article 4(7), never 'Article 4' — and it is the most frequently
    cited article in the instrument, so collapsing it into a single 7,800-character
    record would make its citations useless at exactly the point they matter most.
    Lettered points elsewhere are NOT split: they are governed by a chapeau and are
    not statements of law on their own.
    """
    return paragraph is None and sum(1 for label, _ in points if label.isdigit()) >= 2


def build_records(raw_html: str):
    start = raw_html.find(ENACTING_TERMS_MARKER)
    if start == -1:
        sys.exit("Could not locate the enacting terms; the EUR-Lex markup changed.")
    enacting = raw_html[start:]
    end_match = END_OF_ENACTING_RE.search(enacting)
    if not end_match:
        sys.exit("Could not locate the final formula; the EUR-Lex markup changed.")
    enacting = enacting[:end_match.start()]
    marks = chapter_index(enacting)

    records = []
    for article, article_html in split_articles(enacting):
        num_match = ARTICLE_NUM_RE.search(article_html)
        if num_match and int(num_match.group(1)) != article:
            sys.exit(f"Article id art_{article} contains 'Article {num_match.group(1)}'.")
        title_match = ARTICLE_TITLE_RE.search(article_html)
        title = strip_tags(title_match.group(1)) if title_match else ""
        offset = enacting.find(article_html[:80])
        chapter_num, chapter_title = chapter_at(marks, offset if offset >= 0 else 0)

        for paragraph, text, points in parse_paragraphs(article_html):
            if is_definition_list(paragraph, points):
                units = [
                    (int(label), f"{text} ({label}) {body}".strip(), [])
                    for label, body in points
                ]
            else:
                rendered = " ".join(f"({label}) {ptext}" for label, ptext in points)
                units = [(paragraph, f"{text} {rendered}".strip(), [l for l, _ in points])]

            for unit_paragraph, body, unit_points in units:
                if not body:
                    continue
                records.append(
                {
                    "citation": citation_for(article, unit_paragraph),
                    "article": article,
                    "paragraph": unit_paragraph,
                    "article_title": title,
                    "chapter": chapter_num,
                    "chapter_title": chapter_title,
                    "points": unit_points,
                    "instrument": SHORT_NAME,
                    "instrument_long": LONG_NAME,
                    "celex": CELEX,
                    "source_url": SOURCE_URL,
                    # Two renderings of the same provision, because it is not obvious
                    # which one should be embedded and the sweep in evals/ decides it.
                    # `text` prefixes the citation and article title, on the theory
                    # that a chunk should carry what it is a provision *about*.
                    # `body` is the provision alone. The prefix is shared by every
                    # paragraph of an article, so it may just as easily make siblings
                    # indistinguishable — Article 83 has nine paragraphs that would
                    # all begin "Art. 83(x) — General conditions for imposing
                    # administrative fines".
                    "text": f"{citation_for(article, unit_paragraph)} — {title}. {body}",
                    "body": body,
                }
            )
    return records


def sanity_check(records):
    """Fail loudly on the parse errors that would silently poison retrieval."""
    problems = []
    articles = {r["article"] for r in records}
    missing = sorted(set(range(1, 100)) - articles)
    if missing:
        problems.append(f"missing articles: {missing}")

    # Article 17(1) is the erasure chapeau with points (a)-(f) — a canonical shape.
    art17 = [r for r in records if r["article"] == 17 and r["paragraph"] == 1]
    if not art17:
        problems.append("Article 17(1) not found")
    elif art17[0]["points"] != ["a", "b", "c", "d", "e", "f"]:
        problems.append(f"Article 17(1) points parsed as {art17[0]['points']}")

    # 'controller' is Article 4(7). If the definitions collapsed back into a single
    # Article 4 record, every definition citation silently loses its point number.
    art4_7 = [r for r in records if r["article"] == 4 and r["paragraph"] == 7]
    if not art4_7:
        problems.append("Article 4(7) not found — definitions were not split")
    elif "controller" not in art4_7[0]["text"]:
        problems.append("Article 4(7) does not define 'controller'")

    # The closing article used to absorb the signatures and all 21 footnotes.
    tail = " ".join(r["text"] for r in records if r["article"] == 99)
    for leak in ("Done at Brussels", "OJ L", "document).ready"):
        if leak in tail:
            problems.append(f"Article 99 leaked past the final formula: {leak!r}")

    short = [r["citation"] for r in records if len(r["text"]) < 40]
    if short:
        problems.append(f"suspiciously short records: {short[:5]}")

    untitled = sorted({r["article"] for r in records if not r["article_title"]})
    if untitled:
        problems.append(f"articles with no title: {untitled}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="re-parse the cached HTML instead of downloading")
    args = parser.parse_args()

    raw_html = load_html(args.check)
    records = build_records(raw_html)
    problems = sanity_check(records)
    if problems:
        for problem in problems:
            print(f"  !! {problem}", file=sys.stderr)
        sys.exit("Parse failed its sanity checks; not writing the corpus.")

    with open(OUT_JSONL, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    chars = sum(len(r["text"]) for r in records)
    print(f"{len(records)} provisions from {len({r['article'] for r in records})} articles")
    print(f"{chars:,} characters -> {OUT_JSONL}")


if __name__ == "__main__":
    main()
