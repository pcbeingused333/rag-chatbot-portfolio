"""
The evaluation set: questions whose answering provision is known.

Every question in QUESTIONS is answerable from corpus/gdpr_en.jsonl, and records the
provision that should be retrieved — which is what makes retrieval scoreable without
a human or an LLM in the loop.

Ground truth is a citation, not a substring
-------------------------------------------
The previous version of this file matched distinctive substrings of the correct
passage, and had to warn that a phrase could straddle a chunk boundary and score a
false miss. Holding the ground truth as a provision reference removes that whole
class of error: a chunk boundary can split the wording of Article 33(1), but it
cannot change which provision the chunk came from. It is also the stricter test —
retrieving text that happens to contain the right words is not the same as
retrieving the right authority.

`question_en` is the English form of a non-English question. It exists because the
demo runs an English-only embedding model, and the agent's fix is to translate the
retrieval query rather than pay ~600 MB for a multilingual model. Passing --translate
to the retrieval eval swaps the query for this field, which measures that fix
directly and offline — no LLM call, so the result is free to reproduce.

`reference` is the ground-truth answer, used only by the RAGAS command.

UNANSWERABLE is the other half of the picture, and in this domain the more
important one. See its own note below.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    expect_citations: List[str]
    reference: str
    lang: str = "en"
    question_en: Optional[str] = None

    def retrieval_query(self, translate: bool = False) -> str:
        """The string actually sent to the retriever."""
        if translate and self.question_en:
            return self.question_en
        return self.question


@dataclass(frozen=True)
class Unanswerable:
    """A question the corpus cannot answer, and the reason it cannot."""

    id: str
    question: str
    why: str
    lang: str = "en"
    question_en: Optional[str] = None
    # Provisions that sit near the question without answering it. Retrieving these
    # is correct behaviour; treating them as an answer is not.
    adjacent: List[str] = field(default_factory=list)

    def retrieval_query(self, translate: bool = False) -> str:
        if translate and self.question_en:
            return self.question_en
        return self.question


QUESTIONS: List[Question] = [
    Question(
        id="breach-deadline",
        question="How quickly must a personal data breach be reported to the supervisory authority?",
        expect_citations=["GDPR Art. 33(1)"],
        reference=(
            "Without undue delay and, where feasible, not later than 72 hours after "
            "the controller becomes aware of it."
        ),
    ),
    Question(
        id="breach-notify-subject",
        question="When does a data breach have to be communicated to the affected individuals?",
        expect_citations=["GDPR Art. 34(1)"],
        reference=(
            "When the breach is likely to result in a high risk to the rights and "
            "freedoms of natural persons."
        ),
    ),
    Question(
        id="dsar-deadline",
        question="How long does a controller have to respond to a data subject request?",
        expect_citations=["GDPR Art. 12(3)"],
        reference=(
            "One month from receipt, extendable by two further months where necessary, "
            "taking account of the complexity and number of requests."
        ),
    ),
    Question(
        id="access-scope",
        question="What information can someone obtain when they exercise their right of access?",
        expect_citations=["GDPR Art. 15(1)"],
        reference=(
            "Confirmation of whether their data is processed, access to that data, and "
            "information including the purposes, categories, recipients, retention "
            "period, their rights, the source, and any automated decision-making."
        ),
    ),
    Question(
        id="erasure-grounds",
        question="On what grounds can someone demand that their personal data be erased?",
        expect_citations=["GDPR Art. 17(1)"],
        reference=(
            "Where the data is no longer necessary, consent is withdrawn and there is "
            "no other ground, the subject objects with no overriding grounds, the data "
            "was processed unlawfully, erasure is legally required, or it was collected "
            "for information society services offered to a child."
        ),
    ),
    Question(
        id="portability-format",
        question="In what format must personal data be provided for portability?",
        expect_citations=["GDPR Art. 20(1)"],
        reference=(
            "In a structured, commonly used and machine-readable format."
        ),
    ),
    Question(
        id="lawful-basis",
        question="What are the lawful bases for processing personal data?",
        expect_citations=["GDPR Art. 6(1)"],
        reference=(
            "Consent, performance of a contract, a legal obligation, vital interests, "
            "a public interest or official authority task, and legitimate interests."
        ),
    ),
    Question(
        id="consent-withdrawal",
        question="Can consent be withdrawn, and how easily?",
        expect_citations=["GDPR Art. 7(3)"],
        reference=(
            "Yes, at any time, and it must be as easy to withdraw consent as to give it. "
            "Withdrawal does not affect the lawfulness of processing before it."
        ),
    ),
    Question(
        id="children-consent-age",
        question="At what age can a child consent to information society services on their own?",
        expect_citations=["GDPR Art. 8(1)"],
        reference=(
            "16 years, though Member States may set a lower age, not below 13."
        ),
    ),
    Question(
        id="special-categories",
        question="Which categories of personal data are subject to a general prohibition on processing?",
        expect_citations=["GDPR Art. 9(1)"],
        reference=(
            "Racial or ethnic origin, political opinions, religious or philosophical "
            "beliefs, trade union membership, genetic and biometric data used to "
            "identify a person, health data, and data on sex life or sexual orientation."
        ),
    ),
    Question(
        id="dpo-mandatory",
        question="When is designating a data protection officer mandatory?",
        expect_citations=["GDPR Art. 37(1)"],
        reference=(
            "Where processing is by a public authority, or the core activities require "
            "regular and systematic monitoring of data subjects on a large scale, or "
            "consist of large-scale processing of special categories or criminal data."
        ),
    ),
    Question(
        id="dpia-required",
        question="When is a data protection impact assessment required?",
        expect_citations=["GDPR Art. 35(1)"],
        reference=(
            "Where a type of processing, in particular using new technologies, is likely "
            "to result in a high risk to the rights and freedoms of natural persons."
        ),
    ),
    Question(
        id="records-of-processing",
        question="Who has to keep a record of processing activities?",
        expect_citations=["GDPR Art. 30(1)"],
        reference=(
            "Each controller and, where applicable, its representative, for the "
            "processing under its responsibility."
        ),
    ),
    Question(
        id="fine-higher-tier",
        question="What is the maximum administrative fine for the most serious infringements?",
        expect_citations=["GDPR Art. 83(5)"],
        reference=(
            "Up to EUR 20 000 000, or up to 4 % of total worldwide annual turnover of "
            "the preceding financial year, whichever is higher."
        ),
    ),
    Question(
        id="fine-lower-tier",
        question="What is the fine ceiling for breaching a controller's or processor's obligations?",
        expect_citations=["GDPR Art. 83(4)"],
        reference=(
            "Up to EUR 10 000 000, or up to 2 % of total worldwide annual turnover of "
            "the preceding financial year, whichever is higher."
        ),
    ),
    Question(
        id="automated-decisions",
        question="Is a person entitled to object to a decision made purely by automated means?",
        expect_citations=["GDPR Art. 22(1)"],
        reference=(
            "Yes — the data subject has the right not to be subject to a decision based "
            "solely on automated processing, including profiling, which produces legal "
            "effects concerning them or similarly significantly affects them."
        ),
    ),
    Question(
        id="object-direct-marketing",
        question="Can someone stop their data being used for direct marketing?",
        expect_citations=["GDPR Art. 21(2)", "GDPR Art. 21(3)"],
        reference=(
            "Yes. The data subject may object at any time, and the data must then no "
            "longer be processed for that purpose."
        ),
    ),
    Question(
        id="definition-controller",
        question="What is the definition of a controller?",
        expect_citations=["GDPR Art. 4(7)"],
        reference=(
            "The person or body which, alone or jointly with others, determines the "
            "purposes and means of the processing of personal data."
        ),
    ),
    Question(
        id="definition-processor",
        question="What is a processor?",
        expect_citations=["GDPR Art. 4(8)"],
        reference=(
            "A person or body which processes personal data on behalf of the controller."
        ),
    ),
    Question(
        id="definition-personal-data",
        question="What counts as personal data?",
        expect_citations=["GDPR Art. 4(1)"],
        reference=(
            "Any information relating to an identified or identifiable natural person."
        ),
    ),
    # --- Non-English. These are the reason the translate-the-query fix exists. ---
    Question(
        id="es-breach-deadline",
        question="¿En cuánto tiempo hay que notificar una brecha de seguridad a la autoridad de control?",
        question_en="How quickly must a personal data breach be reported to the supervisory authority?",
        lang="es",
        expect_citations=["GDPR Art. 33(1)"],
        reference=(
            "Sin dilación indebida y, de ser posible, en un plazo máximo de 72 horas "
            "desde que el responsable tenga constancia de ella."
        ),
    ),
    Question(
        id="es-dsar-deadline",
        question="¿Cuánto plazo tengo para responder a una solicitud de acceso?",
        question_en="How long does a controller have to respond to a data subject request?",
        lang="es",
        expect_citations=["GDPR Art. 12(3)"],
        reference="Un mes desde la recepción, prorrogable dos meses más si es necesario.",
    ),
    Question(
        id="es-erasure-grounds",
        question="¿En qué casos puede alguien exigir que se supriman sus datos personales?",
        question_en="On what grounds can someone demand that their personal data be erased?",
        lang="es",
        expect_citations=["GDPR Art. 17(1)"],
        reference=(
            "Cuando los datos ya no son necesarios, se retira el consentimiento sin otra "
            "base, el interesado se opone sin motivos legítimos prevalentes, el "
            "tratamiento fue ilícito, lo exige el Derecho, o se recabaron de un menor."
        ),
    ),
    Question(
        id="es-dpo-mandatory",
        question="¿Cuándo es obligatorio nombrar un delegado de protección de datos?",
        question_en="When is designating a data protection officer mandatory?",
        lang="es",
        expect_citations=["GDPR Art. 37(1)"],
        reference=(
            "Cuando el tratamiento lo lleva a cabo una autoridad pública, o las "
            "actividades principales exigen una observación habitual y sistemática a "
            "gran escala, o el tratamiento a gran escala de categorías especiales."
        ),
    ),
    Question(
        id="fr-fine-higher-tier",
        question="Quel est le montant maximal de l'amende administrative la plus élevée ?",
        question_en="What is the maximum administrative fine for the most serious infringements?",
        lang="fr",
        expect_citations=["GDPR Art. 83(5)"],
        reference=(
            "Jusqu'à 20 000 000 EUR ou 4 % du chiffre d'affaires annuel mondial total, "
            "le montant le plus élevé étant retenu."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Questions the corpus cannot answer.
#
# In this domain this is the failure mode that matters. A retrieval system that
# misses a provision produces a visibly unhelpful answer, and the user goes and
# looks it up. A system that answers a question the source text does not cover —
# from what the model happens to know about data protection law — produces a
# fluent, confident, uncited claim, and nothing about it looks wrong.
#
# These are all adjacent to the GDPR rather than unrelated to it, which is the
# point: the model has read about every one of them, and the corpus contains
# provisions that sit right next to the question without answering it. Article 45
# creates the adequacy mechanism but lists no countries; Article 46 provides for
# standard contractual clauses without reproducing them. The correct answer is to
# retrieve those, notice they do not answer what was asked, and say so.
# ---------------------------------------------------------------------------
UNANSWERABLE: List[Unanswerable] = [
    Unanswerable(
        id="adequacy-country-list",
        question="Which countries have received an adequacy decision?",
        why=(
            "Article 45 establishes the adequacy mechanism and the criteria, but the "
            "decisions themselves are separate Commission acts and no country is named "
            "in the Regulation."
        ),
        adjacent=["GDPR Art. 45(1)", "GDPR Art. 45(2)", "GDPR Art. 45(3)"],
    ),
    Unanswerable(
        id="scc-text",
        question="What is the wording of the standard contractual clauses?",
        why=(
            "Article 46(2) provides for standard contractual clauses as a transfer "
            "safeguard; their text lives in a Commission implementing decision, not in "
            "the Regulation."
        ),
        adjacent=["GDPR Art. 46(2)"],
    ),
    Unanswerable(
        id="schrems-ii",
        question="What did the Court of Justice decide in Schrems II?",
        why="Case law. The Regulation is legislative text and contains no judgments.",
        adjacent=["GDPR Art. 45(1)", "GDPR Art. 46(1)"],
    ),
    Unanswerable(
        id="ccpa-penalty",
        question="What is the maximum penalty under the California Consumer Privacy Act?",
        why="A different instrument in a different jurisdiction; not in this corpus.",
        adjacent=["GDPR Art. 83(5)"],
    ),
    Unanswerable(
        id="uk-gdpr-divergence",
        question="How does the UK GDPR differ from this Regulation after Brexit?",
        why="A separate retained instrument; the Regulation does not describe it.",
        adjacent=["GDPR Art. 3(1)"],
    ),
    Unanswerable(
        id="specific-enforcement-fine",
        question="How much was Meta fined by the Irish Data Protection Commission in 2023?",
        why=(
            "An enforcement decision. Article 83 sets the ceilings and criteria; "
            "individual fines are not in the legislative text."
        ),
        adjacent=["GDPR Art. 83(1)", "GDPR Art. 83(2)"],
    ),
    Unanswerable(
        id="es-adequacy-country-list",
        question="¿Qué países tienen una decisión de adecuación?",
        question_en="Which countries have received an adequacy decision?",
        lang="es",
        why=(
            "El artículo 45 crea el mecanismo, pero las decisiones son actos separados "
            "de la Comisión y el Reglamento no nombra ningún país."
        ),
        adjacent=["GDPR Art. 45(1)", "GDPR Art. 45(3)"],
    ),
]


BY_LANG = {"en", "es", "fr"}


def _filter(items, langs: Optional[List[str]]):
    if not langs:
        return list(items)
    wanted = {lang.lower() for lang in langs}
    unknown = wanted - BY_LANG
    if unknown:
        raise ValueError(f"Unknown language(s): {sorted(unknown)}. Known: {sorted(BY_LANG)}")
    return [item for item in items if item.lang in wanted]


def questions(langs: Optional[List[str]] = None) -> List[Question]:
    """The answerable evaluation set, optionally filtered to specific languages."""
    return _filter(QUESTIONS, langs)


def unanswerable(langs: Optional[List[str]] = None) -> List[Unanswerable]:
    """The questions the corpus cannot answer, optionally filtered by language."""
    return _filter(UNANSWERABLE, langs)
