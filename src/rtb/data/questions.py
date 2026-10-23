"""Two-hop questions: a visible bridge fact, then a leaf fact that holds the answer.

  start --bridge--> bridge entity --leaf--> answer

The bridge hop always lives in a visible profile file. The leaf hop lives in its
own file, which is visible for solvable questions and restricted for impossible
ones. Every (bridge, leaf) relation pair is used for both kinds, so the question
text carries no cue about which kind it is.

Each leaf fact backs at most one question, so an answer never appears as the
answer to some other question in the same world.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from .render import File
from .world import World

# Bridge relations all have subj = start entity and obj = bridge entity.
TEMPLATES: dict[tuple[str, str], list[str]] = {
    ("founded_by", "previously_worked_at"): [
        "Which company did the founder of {start} work for before their current role?",
        "Where was the founder of {start} employed before their current position?",
    ],
    ("founded_by", "studied_at"): [
        "Where did the founder of {start} study?",
        "Which institution did the founder of {start} graduate from?",
    ],
    ("led_by", "previously_worked_at"): [
        "Where did the lead of {start} work before their current role?",
        "Which company employed the person leading {start} before their current job?",
    ],
    ("led_by", "studied_at"): [
        "Where did the lead of {start} study?",
        "Which institution did the person leading {start} graduate from?",
    ],
    ("reports_to", "previously_worked_at"): [
        "Where did {start}'s manager work before their current role?",
        "Which company employed the person {start} reports to before their current job?",
    ],
    ("reports_to", "studied_at"): [
        "Where did {start}'s manager study?",
        "Which institution did the person {start} reports to graduate from?",
    ],
    ("works_at", "acquired"): [
        "Which company was acquired by {start}'s employer?",
        "What company did the organization {start} works for acquire?",
    ],
    ("works_at", "audited_by"): [
        "Which firm audits the company {start} works for?",
        "Who is the external auditor of {start}'s employer?",
    ],
    ("owned_by", "acquired"): [
        "Which company was acquired by the organization that runs {start}?",
        "What company did the owner of {start} acquire?",
    ],
    ("owned_by", "audited_by"): [
        "Which firm audits the organization that runs {start}?",
        "Who is the external auditor of the company behind {start}?",
    ],
}


@dataclass
class Question:
    id: str
    world_id: str
    split: str
    question: str
    answer: str
    aliases: list[str]
    pair: str  # "bridge->leaf"
    chain: list[str]  # [bridge fact id, leaf fact id]
    start: str
    bridge: str
    answer_file: str
    bridge_file: str

    def to_json(self) -> dict:
        return asdict(self)


def build_questions(world: World, files: list[File], split: str, rng: random.Random) -> list[Question]:
    file_of = {fid: f.path for f in files for fid in f.fact_ids}
    leaf_rels = {leaf for _, leaf in TEMPLATES}
    questions: list[Question] = []
    for leaf in world.facts:
        if leaf.rel not in leaf_rels:
            continue
        bridges = [
            b for b in world.facts
            if b.obj == leaf.subj and (b.rel, leaf.rel) in TEMPLATES
        ]
        if not bridges:
            continue  # unreachable leaf facts stay in the corpus as decoys
        b = rng.choice(bridges)
        start, answer = world.entities[b.subj], world.entities[leaf.obj]
        questions.append(Question(
            id=f"{world.id}:q{len(questions)}",
            world_id=world.id,
            split=split,
            question=rng.choice(TEMPLATES[(b.rel, leaf.rel)]).format(start=start.name),
            answer=answer.name,
            aliases=answer.aliases,
            pair=f"{b.rel}->{leaf.rel}",
            chain=[b.id, leaf.id],
            start=start.id,
            bridge=b.obj,
            answer_file=file_of[leaf.id],
            bridge_file=file_of[b.id],
        ))
    return questions
