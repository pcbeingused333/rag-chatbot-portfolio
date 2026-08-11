"""
The evaluation set: questions whose answers are known, against the demo corpus.

Every question is answerable from demo/churreria_calderon.pdf. Each one records
the passage that *should* be retrieved, which is what makes retrieval scoreable
without a human or an LLM in the loop.

`expect` holds distinctive substrings of the correct passage. A retrieved chunk
counts as a hit when it contains any of them (whitespace-normalised, case-
insensitive). Keep the substrings short: the demo splits at 300 characters, so a
long phrase can straddle a chunk boundary and score a false miss.

`question_en` is the English form of a non-English question. It exists because
the demo runs an English-only embedding model, and the agent's fix is to
translate the retrieval query rather than pay ~600 MB for a multilingual model.
Passing --translate to the retrieval eval swaps the query for this field, which
measures that fix directly and offline — no LLM call, so the result is
deterministic and free to reproduce.

`reference` is the ground-truth answer, used only by the RAGAS command.
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    expect: List[str]
    reference: str
    lang: str = "en"
    page: Optional[int] = None  # 1-indexed, for reference in reports
    question_en: Optional[str] = None

    def retrieval_query(self, translate: bool = False) -> str:
        """The string actually sent to the retriever."""
        if translate and self.question_en:
            return self.question_en
        return self.question


QUESTIONS: List[Question] = [
    Question(
        id="hours-monday",
        question="Are you open on Mondays?",
        expect=["Monday: closed", "closed on Mondays"],
        reference="No. Churrería Calderón is closed on Mondays.",
        page=1,
    ),
    Question(
        id="hours-saturday",
        question="What time do you open on Saturday?",
        expect=["Saturday: 8:00 AM"],
        reference="Saturday hours are 8:00 AM to 10:00 PM.",
        page=1,
    ),
    Question(
        id="price-combo",
        question="How much is the churros and chocolate combo?",
        expect=["combo $9.50"],
        reference="The churros + chocolate combo is $9.50 CAD.",
        page=1,
    ),
    Question(
        id="price-churros",
        question="How much does an order of churros cost?",
        expect=["Churros (6 pieces) $6.50"],
        reference="Six churros cost $6.50 CAD.",
        page=1,
    ),
    Question(
        id="catering-minimum",
        question="What is the minimum order for catering?",
        expect=["Minimum order is 50 servings"],
        reference="Catering has a minimum order of 50 servings.",
        page=1,
    ),
    Question(
        id="catering-notice",
        question="How far in advance do I need to book catering?",
        expect=["72 hours notice"],
        reference="Catering needs at least 72 hours notice.",
        page=1,
    ),
    Question(
        id="location",
        question="Where is the shop?",
        expect=["214 Augusta Avenue"],
        reference=(
            "At 214 Augusta Avenue in Kensington Market, Toronto — a 6-minute "
            "walk from Spadina station."
        ),
        page=1,
    ),
    Question(
        id="diet-glutenfree",
        question="Do you have gluten-free churros?",
        expect=["Gluten-free churros are available on Saturdays"],
        reference=(
            "Yes, on Saturdays only, made on a separate line, at $7.50 per six."
        ),
        page=1,
    ),
    Question(
        id="diet-nuts",
        question="Is the food safe for a nut allergy?",
        expect=["cannot guarantee a nut-free product", "kitchen handles nuts"],
        reference=(
            "No. The kitchen handles nuts, so a nut-free product cannot be "
            "guaranteed."
        ),
        page=1,
    ),
    Question(
        id="payment-applepay",
        question="Can I pay with Apple Pay?",
        expect=["Apple Pay"],
        reference="Yes — cash, debit, Visa, Mastercard and Apple Pay are accepted.",
        page=2,
    ),
    Question(
        id="delivery-apps",
        question="Are you on Uber Eats or DoorDash?",
        expect=["third-party apps"],
        reference="No, delivery through third-party apps is not offered.",
        page=2,
    ),
    Question(
        id="loyalty",
        question="Do you have a loyalty program?",
        expect=["every 10th churro order is free"],
        reference=(
            "Yes — every 10th churro order is free after signing up with an "
            "email in store."
        ),
        page=2,
    ),
    # --- Non-English. These are the reason the translate-the-query fix exists. ---
    Question(
        id="es-hours-monday",
        question="¿Abrís los lunes?",
        question_en="Are you open on Mondays?",
        lang="es",
        expect=["Monday: closed", "closed on Mondays"],
        reference="No, los lunes está cerrado.",
        page=1,
    ),
    Question(
        id="es-price-churros",
        question="¿Cuánto cuestan los churros?",
        question_en="How much does an order of churros cost?",
        lang="es",
        expect=["Churros (6 pieces) $6.50"],
        reference="Seis churros cuestan 6,50 $ CAD.",
        page=1,
    ),
    Question(
        id="es-diet-vegan",
        question="¿Tenéis opciones veganas?",
        question_en="Do you have vegan options?",
        lang="es",
        expect=["churros are vegan"],
        reference=(
            "Sí: los churros son veganos si se evita el chocolate con leche y se "
            "elige la opción de bebida de avena."
        ),
        page=2,
    ),
    Question(
        id="es-catering",
        question="¿Hacéis catering para eventos?",
        question_en="Do you do catering for events?",
        lang="es",
        expect=["We cater churro bars", "Minimum order is 50 servings"],
        reference=(
            "Sí, con un mínimo de 50 raciones y 72 horas de antelación; desde "
            "8 $ por persona."
        ),
        page=1,
    ),
    Question(
        id="fr-hours-saturday",
        question="Quels sont vos horaires le samedi ?",
        question_en="What time do you open on Saturday?",
        lang="fr",
        expect=["Saturday: 8:00 AM"],
        reference="Le samedi, de 8h00 à 22h00.",
        page=1,
    ),
]

BY_LANG = {"en", "es", "fr"}


def questions(langs: Optional[List[str]] = None) -> List[Question]:
    """The evaluation set, optionally filtered to specific languages."""
    if not langs:
        return list(QUESTIONS)
    wanted = {lang.lower() for lang in langs}
    unknown = wanted - BY_LANG
    if unknown:
        raise ValueError(f"Unknown language(s): {sorted(unknown)}. Known: {sorted(BY_LANG)}")
    return [q for q in QUESTIONS if q.lang in wanted]
