"""Fetch job postings into data/raw/ and data/postings.jsonl.

Each argument is a job board or a single posting:
  getyourguide                                  bare board name; tries Greenhouse, Lever, Ashby
  https://job-boards.greenhouse.io/<board>      Greenhouse board; add /jobs/<id> for one posting
  https://jobs.eu.lever.co/<company>            Lever board; add /<id> for one posting
  https://jobs.ashbyhq.com/<board>              Ashby board; add /<id> for one posting
  https://careers.example.com                   Teamtailor board (read from /jobs.json)
  https://jobs.zalando.com/en/jobs/<id>         Zalando posting (no JSON-LD; uses the prose article)
  https://bolt.eu/en/careers/positions/<id>     Bolt posting (no JSON-LD; uses the Next.js payload)
  any other posting URL                         page's JobPosting JSON-LD, else the Attrax layout

Boards are filtered to AI/DS titles and capped per company; single URLs are always added.
"""

import html
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

GREENHOUSE_API = "https://boards-api.greenhouse.io/v1/boards/{board}/jobs"
KEYWORDS = re.compile(
    r"data scien|machine learning|\bml\b|\bai\b|llm|applied scien|data en|ai prod", re.I
)
MAX_PER_COMPANY = 10  # keep one company from dominating the dataset

RAW = Path("data/raw")
INDEX = Path("data/postings.jsonl")


@dataclass
class Posting:
    url: str
    company: str
    title: str
    location: str
    body: str
    source: str


def html_to_text(content: str) -> str:
    """Some sources escape their HTML, so unescape first, then strip the tags."""
    soup = BeautifulSoup(html.unescape(content), "html.parser")
    text = soup.get_text("\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def company_from_host(host: str) -> str:
    """careers.mapp.com -> mapp; mapp.teamtailor.com -> mapp."""
    labels = host.removeprefix("www.").split(".")
    return labels[0] if host.endswith(".teamtailor.com") else labels[-2]


def get_json(url: str, **params) -> dict | list:
    resp = httpx.get(url, params=params, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    return resp.json()


def get_html(url: str) -> str:
    # Not raise_for_status(): Bolt serves full posting pages with a 404 status.
    resp = httpx.get(url, timeout=30, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
    if resp.status_code != 200:
        print(f"warning: HTTP {resp.status_code} for {url} (role may be closed)", file=sys.stderr)
    return resp.text


# --- Greenhouse -------------------------------------------------------------


def greenhouse_posting(board: str, job: dict) -> Posting:
    return Posting(
        url=job["absolute_url"],
        company=board,
        title=job["title"],
        location=job["location"]["name"],
        body=html_to_text(job["content"]),
        source="greenhouse",
    )


def greenhouse_board(board: str) -> list[Posting]:
    jobs = get_json(GREENHOUSE_API.format(board=board), content="true")["jobs"]
    return [greenhouse_posting(board, job) for job in jobs]


def greenhouse_job(board: str, job_id: str) -> Posting:
    return greenhouse_posting(board, get_json(f"{GREENHOUSE_API.format(board=board)}/{job_id}"))


# --- Lever ------------------------------------------------------------------


def lever_posting(company: str, job: dict) -> Posting:
    # The intro, the headed lists (requirements live here) and the benefits are separate fields.
    parts = [job["description"]]
    parts += [f"<h3>{lst['text']}</h3><ul>{lst['content']}</ul>" for lst in job["lists"]]
    parts.append(job["additional"])
    return Posting(
        url=job["hostedUrl"],
        company=company,
        title=job["text"],
        location=job["categories"].get("location", ""),
        body=html_to_text("\n".join(parts)),
        source="lever",
    )


def lever_api(host: str, company: str) -> str:
    api_host = "api.eu.lever.co" if host.startswith("jobs.eu.") else "api.lever.co"
    return f"https://{api_host}/v0/postings/{company}"


def lever_board(host: str, company: str) -> list[Posting]:
    jobs = get_json(lever_api(host, company), mode="json")
    return [lever_posting(company, job) for job in jobs]


def lever_job(host: str, company: str, job_id: str) -> Posting:
    return lever_posting(company, get_json(f"{lever_api(host, company)}/{job_id}"))


# --- Ashby ------------------------------------------------------------------


def ashby_posting(board: str, job: dict) -> Posting:
    locations = [job["location"]] + [loc["location"] for loc in job.get("secondaryLocations", [])]
    return Posting(
        url=job["jobUrl"],
        company=board,
        title=job["title"],
        location="; ".join(dict.fromkeys(locations)),
        body=html_to_text(job["descriptionHtml"]),
        source="ashby",
    )


def ashby_board(board: str) -> list[Posting]:
    jobs = get_json(f"https://api.ashbyhq.com/posting-api/job-board/{board}")["jobs"]
    return [ashby_posting(board, job) for job in jobs]


def ashby_job(board: str, job_id: str) -> Posting:
    # Ashby has no public single-job endpoint, so pick the job out of its board.
    for posting in ashby_board(board):
        if posting.url.rstrip("/").endswith(job_id):
            return posting
    raise SystemExit(f"job {job_id} is not on the {board!r} Ashby board (it may be closed)")


# --- schema.org JobPosting (Teamtailor and most other career sites) ----------


def jsonld_location(posting: dict) -> str:
    if posting.get("jobLocationType") == "TELECOMMUTE":
        return "Remote"
    places = posting.get("jobLocation") or []
    if isinstance(places, dict):
        places = [places]
    cities = [p.get("address", {}).get("addressLocality") for p in places]
    return "; ".join(dict.fromkeys(c for c in cities if c))


def jsonld_posting(url: str, posting: dict, body_html: str, source: str) -> Posting:
    return Posting(
        url=url,
        company=company_from_host(urlparse(url).netloc),
        title=posting["title"],
        location=jsonld_location(posting),
        body=html_to_text(body_html),
        source=source,
    )


def find_jobposting(raw: str) -> dict | None:
    soup = BeautifulSoup(raw, "html.parser")
    for script in soup.find_all("script", type="application/ld+json"):
        data = json.loads(script.string or "null")
        for item in data if isinstance(data, list) else [data]:
            if isinstance(item, dict) and item.get("@type") == "JobPosting":
                return item
    return None


def page_job(url: str) -> Posting:
    """A posting on a company career site: JSON-LD if the page has it, else a known page layout."""
    raw = get_html(url)
    posting = find_jobposting(raw)
    if posting is not None:
        logo = posting["hiringOrganization"].get("logo", "")
        source = "teamtailor" if "teamtailor" in logo else "jsonld"
        return jsonld_posting(url, posting, posting["description"], source)
    if "attrax-" in raw:
        return attrax_job(url, raw)
    raise ValueError(f"no JobPosting JSON-LD on {url}; this site needs its own parser")


# --- Attrax (e.g. careers.deliveryhero.com) ----------------------------------


def attrax_job(url: str, raw: str) -> Posting:
    """Attrax pages are server-rendered, with the description in a single description widget."""
    soup = BeautifulSoup(raw, "html.parser")
    # Closed roles keep their text but get a banner inside the description; drop it and warn.
    for banner in soup.select(".attrax-expired-vacancy-message"):
        print(f"warning: {url} is marked as expired", file=sys.stderr)
        banner.decompose()
    location = ""
    for name in soup.select(".attrax-job-information-widget__freetext-field-name"):
        if "LocationText" in name.get_text():
            value = name.find_next(class_="attrax-job-information-widget__freetext-field-value")
            location = value.get_text(" ", strip=True)
            break
    return Posting(
        url=url,
        company=company_from_host(urlparse(url).netloc),
        title=soup.find("h1").get_text(" ", strip=True),
        location=location,
        body=html_to_text(soup.select_one(".description-widget").decode_contents()),
        source="attrax",
    )


# --- Zalando ----------------------------------------------------------------


def zalando_job(url: str) -> Posting:
    """jobs.zalando.com has no JSON-LD; the description is the page's one prose article."""
    page = urlparse(url)
    url = f"https://{page.netloc}{page.path}"  # drop tracking params like ?gh_src=
    soup = BeautifulSoup(get_html(url), "html.parser")
    labels = list(soup.stripped_strings)
    location = labels[labels.index("Location") + 1] if "Location" in labels else ""
    return Posting(
        url=url,
        company="zalando",
        title=soup.find("h1").get_text(" ", strip=True),
        location=location,
        body=html_to_text(soup.select_one("article.prose").decode_contents()),
        source="zalando",
    )


def teamtailor_board(origin: str) -> list[Posting]:
    feed = get_json(f"{origin}/jobs.json")
    return [
        jsonld_posting(item["url"], item["_jobposting"], item["content_html"], "teamtailor")
        for item in feed["items"]
    ]


# --- Bolt -------------------------------------------------------------------


def bolt_job(url: str) -> Posting:
    """The description is only in the Next.js RSC payload, as a length-prefixed text row."""
    raw = get_html(url)
    pushes = re.findall(r"self\.__next_f\.push\((\[.*?\])\)</script>", raw, re.S)
    pushes = [json.loads(c) for c in pushes]
    rsc = "".join(p[1] for p in pushes if len(p) > 1 and isinstance(p[1], str))
    ref = re.search(r'"position-description".*?"__html":"\$([0-9a-f]+)"', rsc).group(1)
    row = re.search(rf"(?:^|\n){ref}:T([0-9a-f]+),", rsc)
    start = len(rsc[: row.end()].encode())
    body_html = rsc.encode()[start : start + int(row.group(1), 16)].decode()  # length is in bytes

    soup = BeautifulSoup(raw, "html.parser")
    labels = list(soup.stripped_strings)
    location = labels[labels.index("LOCATIONS") + 1] if "LOCATIONS" in labels else ""
    return Posting(
        url=url,
        company="bolt",
        title=soup.find("h1").get_text(strip=True),
        location=location,
        body=html_to_text(body_html),
        source="bolt",
    )


# --- dispatch and saving ----------------------------------------------------


def board_by_name(name: str) -> list[Posting]:
    """A bare name doesn't say which platform hosts the board, so try each until one knows it."""
    attempts = [
        lambda: greenhouse_board(name),
        lambda: lever_board("jobs.lever.co", name),
        lambda: lever_board("jobs.eu.lever.co", name),
        lambda: ashby_board(name),
    ]
    for attempt in attempts:
        try:
            return attempt()
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 404:
                raise
    raise SystemExit(f"no Greenhouse, Lever or Ashby board named {name!r}; pass its URL instead")


def fetch(arg: str) -> tuple[list[Posting], bool]:
    """Return the postings for one argument, and whether it was a whole board."""
    if "://" not in arg:
        return board_by_name(arg), True

    url = urlparse(arg)
    host = url.netloc
    parts = [p for p in url.path.split("/") if p]

    if host.endswith("greenhouse.io"):
        if len(parts) >= 3 and parts[1] == "jobs":
            return [greenhouse_job(parts[0], parts[2])], False
        return greenhouse_board(parts[0]), True
    if host.endswith("lever.co"):
        if len(parts) >= 2:
            return [lever_job(host, parts[0], parts[1])], False
        return lever_board(host, parts[0]), True
    if host == "jobs.ashbyhq.com":
        if len(parts) >= 2:
            return [ashby_job(parts[0], parts[1])], False
        return ashby_board(parts[0]), True
    if host == "jobs.zalando.com" and "jobs" in parts:
        return [zalando_job(arg)], False
    if host.endswith("bolt.eu") and "positions" in parts:
        return [bolt_job(arg)], False
    if parts in ([], ["jobs"]):
        return teamtailor_board(f"{url.scheme}://{host}"), True
    return [page_job(arg)], False


def dedupe_key(company: str, title: str) -> tuple[str, str]:
    return company, " ".join(title.lower().split())


def merge_duplicates(postings: Iterable[Posting]) -> list[Posting]:
    """Boards often list one role per city; keep the first and merge the locations."""
    merged: dict[tuple[str, str], Posting] = {}
    for p in postings:
        key = dedupe_key(p.company, p.title)
        if key not in merged:
            merged[key] = p
        elif p.location and p.location not in merged[key].location.split("; "):
            merged[key].location = f"{merged[key].location}; {p.location}".strip("; ")
    return list(merged.values())


def load_index() -> list[dict]:
    if not INDEX.exists():
        return []
    # The index is also edited by hand, so tolerate blank lines.
    lines = INDEX.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def save(postings: list[Posting], is_board: bool) -> None:
    existing = load_index()
    seen_urls = {p["url"] for p in existing}
    seen_keys = {dedupe_key(p["company"], p["title"]) for p in existing}
    next_num = max((int(p["id"][1:]) for p in existing), default=0) + 1
    added = 0

    RAW.mkdir(parents=True, exist_ok=True)
    # A hand edit can drop the final newline; don't glue the next record onto the last line.
    needs_newline = INDEX.exists() and not INDEX.read_text(encoding="utf-8").endswith("\n")
    with INDEX.open("a", encoding="utf-8") as index:
        if needs_newline:
            index.write("\n")
        for p in merge_duplicates(postings):
            if is_board and (added >= MAX_PER_COMPANY or not KEYWORDS.search(p.title)):
                continue
            if p.url in seen_urls or dedupe_key(p.company, p.title) in seen_keys:
                print("skip (already saved)", p.title)
                continue

            pid = f"p{next_num:03d}"
            text = f"{p.title}\n{p.location}\n\n{p.body}"
            (RAW / f"{pid}.txt").write_text(text, encoding="utf-8")

            record = {
                "id": pid,
                "url": p.url,
                "company": p.company,
                "title": p.title,
                "date": date.today().isoformat(),
                "source": p.source,
            }
            index.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(pid, p.title)
            seen_keys.add(dedupe_key(p.company, p.title))
            next_num += 1
            added += 1


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        save(*fetch(arg))
