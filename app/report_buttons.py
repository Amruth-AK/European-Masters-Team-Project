"""
report_buttons.py — Streamlit download buttons for reports
============================================================
The only report-generation function that depends on Streamlit.
All report building/rendering logic lives in mlcompass.reporting.generator.
"""

from __future__ import annotations

import datetime
from typing import Any

import streamlit as st

from mlcompass.reporting.generator import (
    build_report_data,
    generate_html_report,
    generate_markdown_report,
    generate_pdf_report,
)


def add_report_download_buttons(
    session_state: Any,
    dataset_name: str = "dataset",
    key_suffix: str = "",
    report_stage: str = "validation",
    test_baseline_metrics: dict | None = None,
    test_enhanced_metrics: dict | None = None,
) -> None:
    """
    Drop this call in your Streamlit app to add HTML, Markdown, and PDF
    download buttons.

    Parameters
    ----------
    session_state          : streamlit.session_state
    dataset_name           : label used in the report title and filename
    key_suffix             : unique string per call site (prevents duplicate-key errors)
    report_stage           : "validation" or "test"
    test_baseline_metrics  : pass when report_stage="test"
    test_enhanced_metrics  : pass when report_stage="test"
    """
    report_data = build_report_data(
        session_state,
        dataset_name=dataset_name,
        report_stage=report_stage,
        test_baseline_metrics=test_baseline_metrics,
        test_enhanced_metrics=test_enhanced_metrics,
    )

    html_bytes = generate_html_report(report_data)
    md_bytes   = generate_markdown_report(report_data)

    stage_slug = "test" if report_stage == "test" else "validation"
    safe_name  = dataset_name.replace(" ", "_")[:40]
    ts         = datetime.datetime.now().strftime("%Y%m%d")
    base_fn    = f"report_{safe_name}_{stage_slug}_{ts}"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            label="📄 Download HTML Report",
            data=html_bytes,
            file_name=f"{base_fn}.html",
            mime="text/html",
            help="Self-contained HTML with all charts — open in any browser",
            use_container_width=True,
            key=f"dl_html_report_{key_suffix}",
        )

    with col2:
        st.download_button(
            label="📝 Download Markdown",
            data=md_bytes,
            file_name=f"{base_fn}.md",
            mime="text/markdown",
            help="Markdown version — paste into GitHub, Notion, or Confluence",
            use_container_width=True,
            key=f"dl_md_report_{key_suffix}",
        )

    with col3:
        pdf_bytes = generate_pdf_report(report_data)
        if pdf_bytes:
            st.download_button(
                label="📑 Download PDF",
                data=pdf_bytes,
                file_name=f"{base_fn}.pdf",
                mime="application/pdf",
                help="PDF export via WeasyPrint",
                use_container_width=True,
                key=f"dl_pdf_report_{key_suffix}",
            )
        else:
            st.button(
                "📑 PDF unavailable",
                disabled=True,
                use_container_width=True,
                key=f"dl_pdf_disabled_{key_suffix}",
                help="Run `pip install weasyprint` to enable PDF export",
            )
