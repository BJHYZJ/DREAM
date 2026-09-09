#!/usr/bin/env python3
"""Read-only link/evidence/preservation check for the local website preview.

This does not test browser playback, human visual quality, or public deployment.
"""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references = []
        self.videos = []
        self.scripts = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "video":
            self.videos.append(attrs)
        if tag == "script":
            self.scripts.append(attrs)
        for key in ("href", "src", "poster"):
            if attrs.get(key):
                self.references.append(attrs[key])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--website", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-videos", type=int, required=True)
    parser.add_argument("--expected-houses", type=int, required=True)
    args = parser.parse_args()
    website = args.website.resolve()
    page = website / "simulation/index.html"
    html = page.read_text()
    links = Links()
    links.feed(html)
    checks = []
    audits = []
    for ref in links.references:
        url = urlsplit(ref)
        if url.scheme or url.netloc or not url.path:
            continue
        target = (page.parent / unquote(url.path)).resolve()
        checks.append(dict(reference=ref, exists=target.exists()))
        if target.name in ("record_review.json", "audit.json") and target.is_file():
            data = json.loads(target.read_text())
            passed = (data.get("record_review_passed") if target.name == "record_review.json"
                      else data.get("physical_reexecution_passed"))
            audits.append(dict(reference=ref, passed=bool(passed),
                sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    original = subprocess.check_output(["git", "show", "HEAD:index.html"], cwd=website)
    current = (website / "index.html").read_bytes()
    addition = b'      <a href="simulation/">Simulation preview</a>\n'
    preservation = current.count(addition) == 1 and current.replace(addition, b"", 1) == original
    houses = sorted(set(re.findall(r'(?:ProcTHOR|ArchitecTHOR)-(?:Train|Val|Test)-\d+', html)))
    invariants = dict(all_local_references_exist=all(c["exists"] for c in checks),
        original_homepage_preserved_except_one_navigation_link=preservation,
        expected_video_and_house_counts=len(links.videos) == args.expected_videos and len(houses) == args.expected_houses,
        paired_evidence_reports_pass=len(audits) == 2 * args.expected_videos and all(a["passed"] for a in audits),
        no_autoplay=all("autoplay" not in v for v in links.videos),
        no_simulation_page_scripts=not links.scripts,
        preview_is_noindex='content="noindex,nofollow"' in html)
    report = dict(checks=invariants, local_references=checks, evidence=audits,
        houses=houses, video_count=len(links.videos),
        page_sha256=hashlib.sha256(page.read_bytes()).hexdigest(),
        preview_integrity_passed=all(invariants.values()), browser_visual_check_pending=True,
        public_release_ready=False,
        boundary="Local HTML/link/evidence/preservation checks only; no browser screenshot, playback test, public-link verification or deployment.")
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("local_references", "evidence")}, indent=2))
    if not report["preview_integrity_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
