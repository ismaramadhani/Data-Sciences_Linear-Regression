import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm


# =========================
# EXPORT DATA + STATISTICS TO PDF
# =========================
def export_data_pdf(df, stats, filename="report.pdf"):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    x_margin = 2 * cm
    y = height - 2 * cm

    # ---------- TITLE ----------
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x_margin, y, "Agricultural Production Analysis Report")
    y -= 1.2 * cm

    # ---------- DATA PREVIEW ----------
    c.setFont("Helvetica-Bold", 11)
    c.drawString(x_margin, y, "Data Preview (Top 10 Rows)")
    y -= 0.8 * cm

    c.setFont("Helvetica", 9)
    preview = df.head(10)

    for col in preview.columns:
        c.drawString(x_margin, y, str(col))
        y -= 0.4 * cm

        for val in preview[col]:
            c.drawString(x_margin + 1 * cm, y, str(val))
            y -= 0.35 * cm

        y -= 0.3 * cm
        if y < 3 * cm:
            c.showPage()
            y = height - 2 * cm

    # ---------- SUMMARY STATISTICS ----------
    c.showPage()
    y = height - 2 * cm

    c.setFont("Helvetica-Bold", 11)
    c.drawString(x_margin, y, "Summary Statistics")
    y -= 0.8 * cm

    c.setFont("Helvetica", 9)
    stats_reset = stats.reset_index()

    for _, row in stats_reset.iterrows():
        line = " | ".join(str(v) for v in row.values)
        c.drawString(x_margin, y, line)
        y -= 0.4 * cm

        if y < 3 * cm:
            c.showPage()
            y = height - 2 * cm

    c.save()
    buffer.seek(0)

    st.download_button(
        label="📄 Download PDF Report",
        data=buffer,
        file_name=filename,
        mime="application/pdf"
    )


# =========================
# EXPORT FIGURE TO PDF
# =========================
def export_fig_to_pdf(fig, filename="figure.pdf"):
    from io import BytesIO
    import matplotlib.pyplot as plt

    buffer = BytesIO()

    # ===============================
    # HANDLE JIKA fig ADALAH MODULE
    # ===============================
    if hasattr(fig, "gcf"):  
        # berarti fig = matplotlib.pyplot
        current_fig = fig.gcf()
    else:
        # berarti fig = matplotlib.figure.Figure
        current_fig = fig

    current_fig.savefig(buffer, format="pdf", bbox_inches="tight")
    plt.close(current_fig)
    buffer.seek(0)

    st.download_button(
        label=f"🖼️ Download {filename}",
        data=buffer,
        file_name=filename,
        mime="application/pdf"
    )


def download_csv(df):
    """Exports dataframe as CSV."""
    st.download_button(
        label="Download Data as CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name="cleaned_data.csv",
        mime="text/csv"
    )

# def export_data_pdf(df, summary, filename="report.pdf"):
#     """Exports data and summary statistics as a PDF (implement as per your PDF export logic)."""
#     st.write("Export to PDF is not implemented yet.")

# def export_fig_to_pdf(fig, filename="figure.pdf"):
#     """Exports a figure to PDF (implement PDF export logic)."""
#     st.write("Export figure to PDF is not implemented yet.")
