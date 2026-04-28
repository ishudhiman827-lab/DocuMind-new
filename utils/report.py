from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

def create_pdf(summary, filename="report.pdf"):
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("DocuMind AI Report", styles['Title']))
    content.append(Paragraph("<br/><b>Summary:</b><br/>" + summary, styles['Normal']))

    doc.build(content)
    return filename