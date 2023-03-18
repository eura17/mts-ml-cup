from __future__ import annotations

import re
from typing import Callable, Optional


def clean_url(
    url: str,
    preprocessors: list[Callable[[str], str]],
    protected: Optional[set[str]] = None,
) -> str:
    protected = protected or set()
    for preprocessor in preprocessors:
        if url in protected:
            break
        url = preprocessor(url)
    return url


def decode_from_punycode(url: str) -> str:
    try:
        return bytearray(url, "utf-8").decode("idna")
    except Exception:
        return url


def lower(url: str) -> str:
    return url.lower()


def replace_hyphens_with_dots(url: str) -> str:
    return ".".join(filter(lambda p: p != "", url.split("-")))


def remove_page_accelerator(url: str, accelerators: list[str]) -> str:
    for accelerator in accelerators:
        if url.endswith(accelerator):
            return ".".join(url[:-len(accelerator)].split("-"))
    return url


def remove_char(url: str, chars: list[str]) -> str:
    for char in chars:
        url = url.replace(char, "")
    return url


def save_only_suffix(url: str, suffixes: list[str]) -> str:
    for suffix in suffixes:
        if url.endswith(suffix):
            return suffix
    return url


def remove_first_level_domain(url: str) -> str:
    if url.find(".") == -1:
        return url
    return ".".join(url.split(".")[:-1])


def remove_one_char_domains(url: str) -> str:
    return ".".join(filter(lambda domain: len(domain) > 1, url.split(".")))


def remove_domains(url: str, domains: list[str]) -> str:
    return ".".join(filter(lambda domain: domain not in domains, url.split(".")))


def save_full_entry(url: str, entries: list[str]) -> str:
    for entry in entries:
        if entry in url:
            return entry
    return url


def map_url(url: str, mapping: dict[str, str]) -> str:
    return mapping.get(url, None) or url


def save_regexp(url: str, pattern: re.Pattern = re.compile("[a-zA-Zа-яА-Я]+")) -> str:
    return "".join(part for part in pattern.findall(url))
