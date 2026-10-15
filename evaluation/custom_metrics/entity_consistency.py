from __future__ import annotations

from ._common import ENTITY, STOPWORDS, aggregate


class EntityConsistencyMetric:
    """Per-claim entity consistency.

    Every entity-like token in the answer must be supported by the retrieved
    entity tags, the context, the question or the ground truth. The score is
    the fraction supported, so invented entities lower it (previously any
    single overlap scored 1.0).

    Not applicable (None) when the answer mentions no entity.
    """

    name = "entity_consistency"

    def score_one(
        self,
        answer: str,
        entity_tags: list[list[str]] | list[str] | None = None,
        context: str = "",
        ground_truth: str = "",
        question: str = "",
    ) -> float | None:
        entities = {
            m.group(1).lower()
            for m in ENTITY.finditer(answer or "")
            if m.group(1).lower() not in STOPWORDS
        }
        if not entities:
            return None
        tags: set[str] = set()
        for t in entity_tags or []:
            if isinstance(t, (list, tuple, set)):
                tags.update(str(x).lower() for x in t if x)
            elif t:
                tags.add(str(t).lower())
        haystack = " ".join([context, ground_truth, question]).lower()
        supported = sum(1 for e in entities if e in tags or e in haystack)
        return supported / len(entities)

    def score(
        self,
        answers: list[str],
        entity_tags: list[list[list[str]]],
        contexts: list[str] | None = None,
        ground_truths: list[str] | None = None,
        questions: list[str] | None = None,
    ) -> float | None:
        n = len(answers)
        contexts = contexts or [""] * n
        ground_truths = ground_truths or [""] * n
        questions = questions or [""] * n
        tags = list(entity_tags) + [[]] * (n - len(entity_tags))
        return aggregate(
            [
                self.score_one(a, t, c, g, q)
                for a, t, c, g, q in zip(answers, tags, contexts, ground_truths, questions)
            ]
        )
