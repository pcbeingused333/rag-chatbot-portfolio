"""
Generates demo/churreria_calderon.pdf — a sample small-business knowledge base
used to demo the RAG chatbot end-to-end (menu, hours, FAQ, catering, story).

    python demo/make_demo_pdf.py
"""
import os

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "churreria_calderon.pdf")

styles = getSampleStyleSheet()
h = ParagraphStyle("h", parent=styles["Heading2"], spaceBefore=14, textColor="#8a3b00")
body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10.5, leading=15)

SECTIONS = [
    ("Churrería Calderón — Company Knowledge Base",
     "Churrería Calderón is a family-owned Spanish churrería in Toronto, Canada. "
     "We serve fresh churros, porras, and thick Spanish hot chocolate, plus coffee "
     "and small savory items. This document is the reference used by our AI assistant "
     "to answer customer questions."),
    ("Location & Contact",
     "Address: 214 Augusta Avenue, Kensington Market, Toronto, ON. "
     "Phone: (416) 555-0143. Email: hello@churreriacalderon.com. "
     "Website: churreriacalderon.com. We are inside Kensington Market, a 6-minute walk "
     "from Spadina station."),
    ("Opening Hours",
     "Monday: closed. Tuesday to Thursday: 9:00 AM – 7:00 PM. "
     "Friday: 9:00 AM – 10:00 PM. Saturday: 8:00 AM – 10:00 PM. "
     "Sunday: 8:00 AM – 6:00 PM. On public holidays we usually open 10:00 AM – 4:00 PM; "
     "check our website for exact holiday hours."),
    ("Menu & Prices (CAD)",
     "Churros (6 pieces) $6.50. Porras (thick churros, 4 pieces) $6.00. "
     "Spanish hot chocolate (small) $4.00, (large) $5.50. "
     "Churros + chocolate combo $9.50. Filled churros (dulce de leche, chocolate, "
     "or custard) $2.00 each. Café con leche $3.50. Espresso $3.00. "
     "Bottle of water $2.00. Gluten-free churros are available on Saturdays only, $7.50 per 6."),
    ("Dietary Information",
     "Our standard churros contain wheat flour and are fried in vegetable oil (no animal fat). "
     "They are vegetarian. The hot chocolate is made with dairy milk; oat-milk chocolate is "
     "available on request for +$0.75. Nut warning: our kitchen handles nuts, so we cannot "
     "guarantee a nut-free product. Gluten-free churros are made on Saturdays on a separate line."),
    ("Catering & Events",
     "We cater churro bars for weddings, offices, and parties. Minimum order is 50 servings. "
     "A churro bar with two dipping chocolates starts at $8 per person. We need at least "
     "72 hours notice for catering. Delivery within downtown Toronto is $25; free for orders "
     "over $400. To book, email catering@churreriacalderon.com with your date and headcount."),
    ("Orders, Payment & Policies",
     "We accept cash, debit, Visa, Mastercard, and Apple Pay. Online pickup orders can be "
     "placed on our website up to 24 hours ahead. We do not currently offer delivery through "
     "third-party apps. Large fresh orders are made to order and cannot be refunded once "
     "prepared, but we will always fix any mistake on our end."),
    ("Our Story",
     "Churrería Calderón was started by the Calderón family, who moved from Madrid to Toronto. "
     "The recipe for our porras comes from the family's original churrería near Plaza Mayor. "
     "We opened in Kensington Market to bring authentic Spanish churros to Canada, made fresh "
     "throughout the day rather than prepared in advance."),
    ("Frequently Asked Questions",
     "Do you take reservations? No, seating is first come, first served. "
     "Are you open on Mondays? No, we are closed on Mondays. "
     "Do you have vegan options? The churros are vegan if you skip the milk chocolate and "
     "choose the oat-milk option. Do you ship churros? No, churros are best fresh and we do "
     "not ship. Do you have a loyalty program? Yes — every 10th churro order is free when you "
     "sign up with your email in store."),
]


def build():
    doc = SimpleDocTemplate(OUT, pagesize=LETTER,
                            leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                            topMargin=0.9 * inch, bottomMargin=0.9 * inch)
    flow = []
    for i, (title, text) in enumerate(SECTIONS):
        flow.append(Paragraph(title, styles["Title"] if i == 0 else h))
        flow.append(Spacer(1, 4))
        flow.append(Paragraph(text, body))
        flow.append(Spacer(1, 6))
    doc.build(flow)
    print(f"✅ Demo PDF generado en {OUT}")


if __name__ == "__main__":
    build()
