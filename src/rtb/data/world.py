"""Fictional worlds: entities and facts, the source of truth for the dataset.

A world is a small ecosystem of organizations, people and projects. Facts are
(subj, rel, obj) triples of two kinds:

- core facts connect core entities (who works where, who founded what). They are
  always visible to the agent and serve as the first hop of every question.
- leaf facts point from a core entity to a leaf entity that appears in no other
  fact (where someone studied, which firm audits a company). A leaf fact is the
  second hop of a question; whether its file is visible or restricted decides
  whether the question is solvable.

Because each leaf entity is mentioned only in its own leaf fact, hiding that
fact's file removes every trace of the answer from the visible corpus.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

from . import names

CORE_RELATIONS = {"works_at", "reports_to", "founded_by", "owned_by", "led_by", "headquartered_in", "partner_of"}
LEAF_RELATIONS = {"previously_worked_at", "studied_at", "acquired", "audited_by"}


@dataclass
class Entity:
    id: str
    type: str  # person | org | project | city | school | firm
    name: str
    aliases: list[str]
    core: bool


@dataclass
class Fact:
    id: str
    subj: str
    rel: str
    obj: str
    attrs: dict = field(default_factory=dict)

    @property
    def leaf(self) -> bool:
        return self.rel in LEAF_RELATIONS


@dataclass
class World:
    id: str
    domain: str
    entities: dict[str, Entity]
    facts: list[Fact]

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "domain": self.domain,
            "entities": [asdict(e) for e in self.entities.values()],
            "facts": [asdict(f) for f in self.facts],
        }

    @classmethod
    def from_json(cls, d: dict) -> World:
        return cls(
            id=d["id"],
            domain=d["domain"],
            entities={e["id"]: Entity(**e) for e in d["entities"]},
            facts=[Fact(**f) for f in d["facts"]],
        )

    def facts_about(self, subj: str, rel: str | None = None) -> list[Fact]:
        return [f for f in self.facts if f.subj == subj and (rel is None or f.rel == rel)]


class NameRegistry:
    """Hands out names that are unique across the dataset (full names) and unique
    within a world (brand words, surnames), so every alias of a leaf answer can be
    matched in text without colliding with another entity."""

    def __init__(self) -> None:
        self.full_names: set[str] = set()

    def new_world(self) -> None:
        self.world_tokens: set[str] = set()

    def _claim(self, full: str, tokens: list[str]) -> bool:
        if full in self.full_names or any(t.lower() in self.world_tokens for t in tokens):
            return False
        self.full_names.add(full)
        self.world_tokens.update(t.lower() for t in tokens)
        return True

    def person(self, rng: random.Random) -> tuple[str, list[str]]:
        while True:
            first, last = rng.choice(names.FIRST_NAMES), rng.choice(names.LAST_NAMES)
            full = f"{first} {last}"
            if self._claim(full, [last]):
                return full, [full, f"{first[0]}. {last}"]

    def org(self, rng: random.Random, kind: str) -> tuple[str, list[str]]:
        while True:
            brand = rng.choice(names.BRAND_WORDS)
            if kind == "company":
                full = f"{brand} {rng.choice(names.COMPANY_SUFFIXES)}"
            elif kind == "firm":
                full = f"{brand} {rng.choice(names.FIRM_SUFFIXES)}"
            else:  # school
                full = rng.choice(names.SCHOOL_PATTERNS).format(b=brand)
            if self._claim(full, [brand]):
                return full, [full, brand]

    def project(self, rng: random.Random) -> tuple[str, list[str]]:
        while True:
            word = rng.choice(names.PROJECT_WORDS)
            if word.lower() not in self.world_tokens:
                self.world_tokens.add(word.lower())
                return f"Project {word}", [f"Project {word}", word]


def generate_world(
    world_id: str,
    rng: random.Random,
    registry: NameRegistry,
    n_orgs: tuple[int, int] = (5, 7),
    staff_per_org: tuple[int, int] = (3, 5),
    projects_per_org: tuple[int, int] = (1, 2),
    leaf_probs: dict[str, float] | None = None,
) -> World:
    leaf_probs = leaf_probs or {"previously_worked_at": 0.6, "studied_at": 0.5, "acquired": 0.7, "audited_by": 0.7}
    registry.new_world()
    entities: dict[str, Entity] = {}
    facts: list[Fact] = []

    def add_entity(type_: str, name: str, aliases: list[str], core: bool) -> str:
        eid = f"{world_id}:{type_[0]}{len(entities)}"
        entities[eid] = Entity(eid, type_, name, aliases, core)
        return eid

    def add_fact(subj: str, rel: str, obj: str, **attrs) -> None:
        facts.append(Fact(f"{world_id}:f{len(facts)}", subj, rel, obj, attrs))

    cities = rng.sample(names.CITIES, 4)
    city_ids = [add_entity("city", c, [c], core=True) for c in cities]

    orgs: list[str] = []
    staff: dict[str, list[str]] = {}
    for _ in range(rng.randint(*n_orgs)):
        org = add_entity("org", *registry.org(rng, "company"), core=True)
        orgs.append(org)
        add_fact(org, "headquartered_in", rng.choice(city_ids))
        people = [add_entity("person", *registry.person(rng), core=True) for _ in range(rng.randint(*staff_per_org))]
        staff[org] = people
        founder, rest = people[0], people[1:]
        add_fact(org, "founded_by", founder)
        add_fact(founder, "works_at", org, role="Chief Executive Officer")
        roles = rng.sample(names.ROLES, len(rest))
        for i, (person, role) in enumerate(zip(rest, roles)):
            add_fact(person, "works_at", org, role=role)
            add_fact(person, "reports_to", rng.choice(people[: i + 1]))

    for org in orgs:
        if len(orgs) > 1 and rng.random() < 0.5:
            add_fact(org, "partner_of", rng.choice([o for o in orgs if o != org]))
        for _ in range(rng.randint(*projects_per_org)):
            project = add_entity("project", *registry.project(rng), core=True)
            add_fact(project, "owned_by", org)
            add_fact(project, "led_by", rng.choice(staff[org]))

    # Leaf facts: each creates a fresh entity that appears nowhere else.
    for org in orgs:
        if rng.random() < leaf_probs["acquired"]:
            target = add_entity("org", *registry.org(rng, "company"), core=False)
            add_fact(org, "acquired", target, year=rng.randint(2008, 2024))
        if rng.random() < leaf_probs["audited_by"]:
            add_fact(org, "audited_by", add_entity("firm", *registry.org(rng, "firm"), core=False))
        for person in staff[org]:
            if rng.random() < leaf_probs["previously_worked_at"]:
                prev = add_entity("org", *registry.org(rng, "company"), core=False)
                add_fact(person, "previously_worked_at", prev, years=rng.randint(2, 9))
            if rng.random() < leaf_probs["studied_at"]:
                add_fact(person, "studied_at", add_entity("school", *registry.org(rng, "school"), core=False))

    return World(world_id, "corporate", entities, facts)
