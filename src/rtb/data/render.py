"""Render a world's facts into a directory of realistic-looking files.

Layout per world:
  people/<first>_<last>.md            profile: role, employer, manager (core)
  orgs/<brand>.md                     overview: founder, HQ, partners (core)
  projects/<word>.md                  owner, lead (core)
  hr/<last>_<first>_employment.md     one leaf fact each, so the file can be moved
  hr/<last>_<first>_education.md        to restricted/ without touching anything else
  corporate/<brand>_acquisitions.md
  finance/<brand>_audit.md
  notes/<date>_<slug>.md              distractor meeting notes (core entities only)

Rendering goes through a Writer. TemplateWriter is deterministic and needs no API;
an LLM writer with the same interface can replace it for more natural prose.
"""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass
from typing import Protocol

from .world import Entity, Fact, World


@dataclass
class File:
    path: str
    text: str
    fact_ids: list[str]
    kind: str  # profile | leaf | note

    def to_json(self) -> dict:
        return asdict(self)


class Writer(Protocol):
    def write(self, doc_type: str, facts: list[Fact], world: World, rng: random.Random) -> str: ...


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


FILLER = [
    "This page is maintained by the internal knowledge team and reviewed quarterly.",
    "For corrections, file a ticket with the records office.",
    "Last updated during the annual records audit.",
    "Some details on this page are summarized from onboarding paperwork.",
    "See the shared drive for older revisions of this document.",
    "Questions about this record should go to the operations desk.",
]


class TemplateWriter:
    """Placeholder writer: fixed sentence templates plus filler."""

    def write(self, doc_type: str, facts: list[Fact], world: World, rng: random.Random) -> str:
        e = world.entities
        n = lambda eid: e[eid].name  # noqa: E731
        lines: list[str] = []
        if doc_type == "person_profile":
            subj = facts[0].subj
            lines.append(f"# {n(subj)}")
            for f in facts:
                if f.rel == "works_at":
                    lines.append(rng.choice([
                        f"{n(subj)} is the {f.attrs['role']} at {n(f.obj)}.",
                        f"{n(subj)} works at {n(f.obj)} as {f.attrs['role']}.",
                    ]))
                elif f.rel == "reports_to":
                    lines.append(rng.choice([
                        f"{n(subj)} reports to {n(f.obj)}.",
                        f"Manager: {n(f.obj)}.",
                    ]))
        elif doc_type == "org_overview":
            subj = facts[0].subj
            lines.append(f"# {n(subj)}")
            for f in facts:
                if f.rel == "founded_by":
                    lines.append(rng.choice([f"{n(subj)} was founded by {n(f.obj)}.", f"Founder: {n(f.obj)}."]))
                elif f.rel == "headquartered_in":
                    lines.append(f"The company is headquartered in {n(f.obj)}.")
                elif f.rel == "partner_of":
                    lines.append(f"{n(subj)} has a long-standing partnership with {n(f.obj)}.")
        elif doc_type == "project_page":
            subj = facts[0].subj
            lines.append(f"# {n(subj)}")
            for f in facts:
                if f.rel == "owned_by":
                    lines.append(f"{n(subj)} is run by {n(f.obj)}.")
                elif f.rel == "led_by":
                    lines.append(rng.choice([f"Project lead: {n(f.obj)}.", f"The project is led by {n(f.obj)}."]))
        elif doc_type == "leaf":
            (f,) = facts
            s, o = n(f.subj), n(f.obj)
            if f.rel == "previously_worked_at":
                lines += [f"# Employment history: {s}", rng.choice([
                    f"Before the current role, {s} spent {f.attrs['years']} years at {o}.",
                    f"Prior employer: {o} ({f.attrs['years']} years).",
                ])]
            elif f.rel == "studied_at":
                lines += [f"# Education record: {s}", rng.choice([
                    f"{s} graduated from {o}.",
                    f"Degree awarded by {o}.",
                ])]
            elif f.rel == "acquired":
                lines += [f"# Acquisitions: {s}", f"In {f.attrs['year']}, {s} completed its acquisition of {o}."]
            elif f.rel == "audited_by":
                lines += [f"# Audit engagement: {s}", rng.choice([
                    f"The external auditor of {s} is {o}.",
                    f"{s} has engaged {o} as its independent auditor.",
                ])]
        else:
            raise ValueError(doc_type)
        lines.append(rng.choice(FILLER))
        return "\n\n".join(lines) + "\n"


def _note(world: World, rng: random.Random, idx: int) -> File:
    people = [x for x in world.entities.values() if x.type == "person"]
    projects = [x for x in world.entities.values() if x.type == "project"]
    attendees = rng.sample(people, min(len(people), rng.randint(2, 4)))
    project = rng.choice(projects)
    topic = rng.choice(["timeline review", "budget sync", "staffing check-in", "vendor update", "risk review"])
    date = f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    text = "\n\n".join([
        f"# {project.name} {topic}",
        f"Date: {date}",
        "Attendees: " + ", ".join(a.name for a in attendees),
        rng.choice([
            f"{attendees[0].name} walked through open items for {project.name}.",
            f"The group agreed to revisit the {project.name} schedule next month.",
            f"{attendees[-1].name} will circulate a revised plan for {project.name}.",
        ]),
        "Action items were recorded in the tracker.",
    ]) + "\n"
    return File(f"notes/{date}_{slug(topic)}_{idx}.md", text, [], "note")


def _leaf_path(f: Fact, world: World) -> str:
    subj: Entity = world.entities[f.subj]
    if subj.type == "person":
        first, last = subj.name.split(" ", 1)
        stem = f"{slug(last)}_{slug(first)}"
        return f"hr/{stem}_{'employment' if f.rel == 'previously_worked_at' else 'education'}.md"
    brand = slug(subj.aliases[1])
    return f"corporate/{brand}_acquisitions.md" if f.rel == "acquired" else f"finance/{brand}_audit.md"


def render_world(world: World, rng: random.Random, writer: Writer | None = None, n_notes: tuple[int, int] = (8, 14)) -> list[File]:
    writer = writer or TemplateWriter()
    files: list[File] = []
    for ent in world.entities.values():
        if not ent.core or ent.type == "city":
            continue
        core = [f for f in world.facts_about(ent.id) if not f.leaf]
        if ent.type == "person":
            first, last = ent.name.split(" ", 1)
            path, doc_type = f"people/{slug(first)}_{slug(last)}.md", "person_profile"
        elif ent.type == "org":
            path, doc_type = f"orgs/{slug(ent.aliases[1])}.md", "org_overview"
        else:
            path, doc_type = f"projects/{slug(ent.aliases[1])}.md", "project_page"
        files.append(File(path, writer.write(doc_type, core, world, rng), [f.id for f in core], "profile"))
    for f in world.facts:
        if f.leaf:
            files.append(File(_leaf_path(f, world), writer.write("leaf", [f], world, rng), [f.id], "leaf"))
    files += [_note(world, rng, i) for i in range(rng.randint(*n_notes))]
    return files
