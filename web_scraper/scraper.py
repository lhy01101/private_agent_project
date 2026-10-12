"""
通用网页正文提取工具

用法:
    python scraper.py <url> [--output json|text|markdown] [--save <file>]

示例:
    python scraper.py https://example.com
    python scraper.py https://example.com --output markdown --save result.md
"""

import argparse
import json
import sys
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Comment

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

TAG_WEIGHT = {
    "article": 3,
    "main": 3,
    "section": 2,
    "div": 1,
    "p": 3,
    "pre": 2,
    "code": 2,
    "blockquote": 2,
    "h1": 0,
    "h2": 0,
    "h3": 0,
    "h4": 0,
    "h5": 0,
    "h6": 0,
}

NOISE_TAGS = {
    "script", "style", "nav", "footer", "header",
    "aside", "noscript", "iframe", "svg", "form",
    "button", "input", "select", "textarea",
}

NOISE_CLASSES = {
    "sidebar", "nav", "menu", "footer", "header",
    "comment", "comments", "advertisement", "ad",
    "social", "share", "related", "recommend",
}


def fetch_page(url: str, timeout: int = 15) -> str:
    resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


def clean_soup(soup: BeautifulSoup) -> BeautifulSoup:
    for tag in soup.find_all(NOISE_TAGS):
        tag.decompose()
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()
    for tag in soup.find_all(True):
        classes = " ".join(tag.get("class", []))
        tag_id = tag.get("id", "")
        combined = (classes + " " + tag_id).lower()
        if any(kw in combined for kw in NOISE_CLASSES):
            tag.decompose()
    return soup


def extract_main_content(soup: BeautifulSoup):
    candidates = []
    for tag_name, weight in TAG_WEIGHT.items():
        if weight <= 0:
            continue
        for el in soup.find_all(tag_name):
            text = el.get_text(separator=" ", strip=True)
            text_len = len(text)
            if text_len < 50:
                continue
            score = weight * min(text_len, 2000) / 100
            p_count = len(el.find_all("p"))
            score += p_count * 3
            link_text = sum(
                len(a.get_text(strip=True)) for a in el.find_all("a")
            )
            if text_len > 0:
                link_ratio = link_text / text_len
                if link_ratio > 0.6:
                    score *= 0.3
            candidates.append((score, el))

    if not candidates:
        body = soup.find("body")
        return body if body else soup

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def extract_page(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    title = ""
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)

    meta_desc = ""
    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        meta_desc = meta["content"].strip()

    clean_soup(soup)

    content_el = extract_main_content(soup)

    headings = []
    for h in content_el.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        headings.append({"level": int(h.name[1]), "text": h.get_text(strip=True)})

    full_text = content_el.get_text(separator="\n", strip=True)

    paragraphs = []
    text_tags = {"p", "blockquote", "pre", "li"}
    if content_el.name in text_tags:
        text = content_el.get_text(strip=True)
        if text:
            paragraphs.append(text)
    for p in content_el.find_all(list(text_tags)):
        text = p.get_text(strip=True)
        if text:
            paragraphs.append(text)
    if not paragraphs and full_text:
        paragraphs = [line.strip() for line in full_text.splitlines() if line.strip()]

    links = []
    for a in content_el.find_all("a", href=True):
        href = urljoin(url, a["href"])
        link_text = a.get_text(strip=True)
        if link_text and href.startswith("http"):
            links.append({"text": link_text, "url": href})

    images = []
    for img in content_el.find_all("img", src=True):
        src = urljoin(url, img["src"])
        alt = img.get("alt", "")
        images.append({"src": src, "alt": alt})

    return {
        "url": url,
        "domain": urlparse(url).netloc,
        "title": title,
        "description": meta_desc,
        "headings": headings,
        "paragraphs": paragraphs,
        "links": links,
        "images": images,
        "full_text": full_text,
        "text_length": len(full_text),
    }


def to_markdown(data: dict) -> str:
    lines = [f"# {data['title']}", "", f"> 来源: {data['url']}", ""]
    if data["description"]:
        lines += [f"> {data['description']}", ""]
    for h in data["headings"]:
        prefix = "#" * min(h["level"] + 1, 6)
        lines.append(f"{prefix} {h['text']}")
        lines.append("")
    for p in data["paragraphs"]:
        lines.append(p)
        lines.append("")
    if data["links"]:
        lines.append("## 链接")
        lines.append("")
        for link in data["links"][:30]:
            lines.append(f"- [{link['text']}]({link['url']})")
        lines.append("")
    return "\n".join(lines)


def to_text(data: dict) -> str:
    lines = [f"标题: {data['title']}", f"来源: {data['url']}", ""]
    for p in data["paragraphs"]:
        lines.append(p)
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="通用网页正文提取工具")
    parser.add_argument("url", help="要爬取的网页 URL")
    parser.add_argument(
        "--output", "-o",
        choices=["json", "text", "markdown"],
        default="text",
        help="输出格式 (默认: text)",
    )
    parser.add_argument("--save", "-s", help="保存到文件")
    args = parser.parse_args()

    url = args.url
    if not url.startswith("http"):
        url = "https://" + url

    print(f"正在爬取: {url}", file=sys.stderr)
    html = fetch_page(url)
    print(f"页面大小: {len(html)} 字节", file=sys.stderr)

    data = extract_page(html, url)
    print(f"提取正文: {data['text_length']} 字符", file=sys.stderr)

    if args.output == "json":
        result = json.dumps(data, ensure_ascii=False, indent=2)
    elif args.output == "markdown":
        result = to_markdown(data)
    else:
        result = to_text(data)

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"已保存到: {args.save}", file=sys.stderr)
    else:
        print(result)


if __name__ == "__main__":
    main()
