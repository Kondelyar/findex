import re
from dataclasses import dataclass

from findex.models import Index
from findex.tokenize import tokenize


class QueryError(ValueError):
    pass


class Node:
    def __and__(self, other: "Node") -> "And":
        return And(self, other)

    def __or__(self, other: "Node") -> "Or":
        return Or(self, other)

    def __invert__(self) -> "Not":
        return Not(self)


@dataclass(frozen=True)
class Term(Node):
    text: str

    def evaluate(self, index: Index) -> set[int]:
        return {p.doc_id for p in index.get(self.text, ())}

    def positive_terms(self) -> list[str]:
        return [self.text]


@dataclass(frozen=True)
class Phrase(Node):
    terms: tuple[str, ...]

    def evaluate(self, index: Index) -> set[int]:
        if not index.has_positions:
            raise QueryError("phrase queries need an index built with --positions")
        by_doc = [{p.doc_id: p.positions for p in index.get(t, ())} for t in self.terms]
        candidates = set.intersection(*(set(d) for d in by_doc))
        return {doc for doc in candidates if _is_phrase([d[doc] for d in by_doc])}

    def positive_terms(self) -> list[str]:
        return list(self.terms)


def _is_phrase(position_lists: list[tuple[int, ...]]) -> bool:
    starts = set(position_lists[0])
    for offset, positions in enumerate(position_lists[1:], start=1):
        starts &= {p - offset for p in positions}
    return bool(starts)


@dataclass(frozen=True)
class And(Node):
    left: Node
    right: Node

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) & self.right.evaluate(index)

    def positive_terms(self) -> list[str]:
        return self.left.positive_terms() + self.right.positive_terms()


@dataclass(frozen=True)
class Or(Node):
    left: Node
    right: Node

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) | self.right.evaluate(index)

    def positive_terms(self) -> list[str]:
        return self.left.positive_terms() + self.right.positive_terms()


@dataclass(frozen=True)
class Not(Node):
    child: Node

    def evaluate(self, index: Index) -> set[int]:
        return set(index.doc_lengths) - self.child.evaluate(index)

    def positive_terms(self) -> list[str]:
        return []


_TOKEN_RE = re.compile(r'"[^"]*"|[()]|[^\s()"]+')
_OPERATORS = ("(", ")", "AND", "OR", "NOT")


def _lex(query: str) -> list[str | Node]:
    tokens: list[str | Node] = []
    for raw in _TOKEN_RE.findall(query):
        if raw in _OPERATORS:
            tokens.append(raw)
        elif raw.startswith('"'):
            terms = tuple(tokenize(raw.strip('"')))
            if terms:
                tokens.append(Phrase(terms))
        else:
            terms = tuple(tokenize(raw))
            if len(terms) == 1:
                tokens.append(Term(terms[0]))
            elif terms:
                tokens.append(Phrase(terms))
    return tokens


class _Parser:
    def __init__(self, tokens: list[str | Node]):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self):
        tok = self.peek()
        self.pos += 1
        return tok

    def parse(self) -> Node:
        if not self.tokens:
            raise QueryError("empty query")
        node = self.or_expr()
        if self.peek() is not None:
            raise QueryError(f"unexpected {self.peek()!r}")
        return node

    def or_expr(self) -> Node:
        node = self.and_expr()
        while self.peek() == "OR":
            self.take()
            node = Or(node, self.and_expr())
        return node

    def and_expr(self) -> Node:
        node = self.not_expr()
        while self.peek() not in (None, "OR", ")"):
            if self.peek() == "AND":
                self.take()
            node = And(node, self.not_expr())
        return node

    def not_expr(self) -> Node:
        if self.peek() == "NOT":
            self.take()
            return Not(self.not_expr())
        return self.atom()

    def atom(self) -> Node:
        tok = self.take()
        if isinstance(tok, Node):
            return tok
        if tok == "(":
            node = self.or_expr()
            if self.take() != ")":
                raise QueryError("missing ')'")
            return node
        raise QueryError("unexpected end of query" if tok is None else f"unexpected {tok!r}")


def parse(query: str) -> Node:
    return _Parser(_lex(query)).parse()
