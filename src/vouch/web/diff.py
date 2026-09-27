"""Highlight for the review page: words the tailored text uses that the profile's wording didn't.

Reordering isn't highlighted, only new wording, because new wording is what needs checking."""

import re

from markupsafe import Markup, escape

# Function words carry no claim; highlighting them only adds noise.
_STOPWORDS = set(
    "a an and as at by for from in into of on or the to with via using its their this that".split()
)


def _key(word: str) -> str:
    return re.sub(r"[^\w+#/.-]", "", word.lower()).strip(".-")


def highlight_changes(original: str, new: str) -> Markup:
    """`new` as HTML, with words that don't appear in `original` wrapped in <mark>."""
    known = {_key(w) for w in original.split()}
    out = []
    for token in re.findall(r"\S+|\s+", new):
        key = _key(token)
        if token.isspace() or not key or key in known or key in _STOPWORDS:
            out.append(str(escape(token)))
        else:
            out.append(f"<mark>{escape(token)}</mark>")
    # Merge adjacent highlights so a new phrase reads as one stroke.
    return Markup(re.sub(r"</mark>(\s+)<mark>", r"\1", "".join(out)))
