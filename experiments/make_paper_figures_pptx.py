"""Build the paper figure deck (editable PowerPoint) from the results CSVs.

Each slide is one paper figure, drawn with native PPT shapes/charts so co-authors
can restyle and export to PDF/EMF for LaTeX. Data comes from
experiments/motif_check_multimotif_results.csv — re-run this script after adding
rows (e.g. the 4x4 model or ablations) to refresh the charts.

    python experiments/make_paper_figures_pptx.py
Writes experiments/paper_figures.pptx
"""
import csv, os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "motif_check_multimotif_results.csv")
OUT = os.path.join(HERE, "paper_figures.pptx")

MODELS = ["irish20k_evalfix", "irish50k", "irish50k_r2", "irishfull"]
MODEL_LABEL = {"irish20k_evalfix": "20k", "irish50k": "50k", "irish50k_r2": "50k (r2)",
               "irishfull": "216k (full)"}
MOTIF_ORDER = ["0,-1,-1,-1", "0,1,1,1", "0,1,-1,-1", "0,-1,-1,1", "0,1,1,-1",
               "0,3,-1,-1", "0,1,-1,1", "0,-1,-2,2", "0,2,-2,-1", "0,3,3,3", "0,-3,-3,-3"]

ACCENT = RGBColor(0x2E, 0x5E, 0xAA)
GRAY = RGBColor(0x88, 0x88, 0x88)


def load():
    rows = list(csv.DictReader(open(CSV)))
    body = {}
    for r in rows:
        if r["body_n"] != "":
            body[(r["model"], r["motif"])] = int(r["body_n"])
    return body


def title_slide(prs, title, sub):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    tb = s.shapes.add_textbox(Inches(0.5), Inches(0.35), Inches(12.3), Inches(0.8)).text_frame
    tb.text = title
    tb.paragraphs[0].font.size = Pt(24); tb.paragraphs[0].font.bold = True
    if sub:
        tb2 = s.shapes.add_textbox(Inches(0.5), Inches(1.0), Inches(12.3), Inches(0.5)).text_frame
        tb2.text = sub
        tb2.paragraphs[0].font.size = Pt(12); tb2.paragraphs[0].font.color.rgb = GRAY
    return s


def add_chart(slide, chart_type, chart_data, x, y, cx, cy, y_title=None):
    gframe = slide.shapes.add_chart(chart_type, Inches(x), Inches(y), Inches(cx), Inches(cy), chart_data)
    ch = gframe.chart
    ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.BOTTOM
    ch.legend.include_in_layout = False
    try:
        ch.value_axis.has_major_gridlines = True
        if y_title:
            ch.value_axis.axis_title.text_frame.text = y_title
    except Exception:
        pass
    return ch


def main():
    body = load()
    prs = Presentation()
    prs.slide_width = Inches(13.33); prs.slide_height = Inches(7.5)

    # ---- Slide 1: method overview schematic (fig:overview) ----
    # Three-stage pipeline: Data Preparation -> Training -> Inference/Generation.
    # Data artifacts = folded-corner shapes; processes = rounded rectangles; each
    # stage gets a colored header ribbon and a tinted panel. All native shapes.
    s = title_slide(prs, "Fig 1 — System overview (fig:overview)",
                    "Editable schematic: restyle in PowerPoint, export PDF/EMF for LaTeX")

    from pptx.oxml.ns import qn
    from pptx.enum.text import PP_ALIGN

    STAGES = {  # (header color, panel tint)
        "prep":  (RGBColor(0x2E, 0x5E, 0xAA), RGBColor(0xEC, 0xF2, 0xFA)),
        "train": (RGBColor(0x2F, 0x7D, 0x4F), RGBColor(0xEB, 0xF5, 0xEE)),
        "infer": (RGBColor(0xC2, 0x61, 0x1E), RGBColor(0xFB, 0xF0, 0xE6)),
    }
    DARK = RGBColor(0x33, 0x33, 0x33)
    WHITE = RGBColor(0xFF, 0xFF, 0xFF)

    def _tail_arrow(conn):
        ln = conn.line._get_or_add_ln()
        tail = ln.makeelement(qn('a:tailEnd'), {'type': 'triangle', 'w': 'med', 'len': 'med'})
        ln.append(tail)

    def arrow(x1, y1, x2, y2, color=DARK, weight=1.75, dash=None):
        c = s.shapes.add_connector(2, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
        c.line.width = Pt(weight); c.line.color.rgb = color
        if dash:
            ln = c.line._get_or_add_ln()
            d = ln.makeelement(qn('a:prstDash'), {'val': dash}); ln.append(d)
        _tail_arrow(c)
        return c

    def panel(x, w, key, label):
        hdr_c, tint = STAGES[key]
        bg = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(1.35), Inches(w), Inches(5.85))
        bg.fill.solid(); bg.fill.fore_color.rgb = tint
        bg.line.color.rgb = hdr_c; bg.line.width = Pt(1.0)
        bg.shadow.inherit = False
        hdr = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x + 0.12), Inches(1.5), Inches(w - 0.24), Inches(0.42))
        hdr.fill.solid(); hdr.fill.fore_color.rgb = hdr_c; hdr.line.fill.background()
        tf = hdr.text_frame; tf.text = label
        tf.paragraphs[0].font.size = Pt(13); tf.paragraphs[0].font.bold = True
        tf.paragraphs[0].font.color.rgb = WHITE; tf.paragraphs[0].alignment = PP_ALIGN.CENTER

    def node(x, y, w, h, title, body, key, shape=MSO_SHAPE.ROUNDED_RECTANGLE, mono_body=False):
        hdr_c, _ = STAGES[key]
        sh = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
        sh.fill.solid(); sh.fill.fore_color.rgb = WHITE
        sh.line.color.rgb = hdr_c; sh.line.width = Pt(1.25)
        sh.shadow.inherit = False
        tf = sh.text_frame; tf.word_wrap = True
        tf.margin_top = Emu(27432); tf.margin_bottom = Emu(27432)
        tf.text = title
        p0 = tf.paragraphs[0]; p0.font.size = Pt(10.5); p0.font.bold = True
        p0.font.color.rgb = hdr_c; p0.alignment = PP_ALIGN.CENTER
        for line in body.split("\n"):
            p = tf.add_paragraph(); p.text = line
            p.font.size = Pt(8.5); p.font.color.rgb = DARK; p.alignment = PP_ALIGN.CENTER
            if mono_body: p.font.name = "Consolas"
        return sh

    # ---------- Panel 1: Data Preparation ----------
    panel(0.35, 4.15, "prep", "1 · Data Preparation")
    node(0.60, 2.10, 3.65, 0.78, "Score corpora",
         "OpenScore Lieder (1.3k pieces) · Irishman folk tunes (216k)",
         "prep", MSO_SHAPE.FOLDED_CORNER)
    node(0.60, 3.16, 3.65, 0.78, "Preprocessing",
         "V:1 melody wrap · rest-bar omission · 15-key transposition augmentation",
         "prep")
    node(0.60, 4.22, 3.65, 0.94, "Motif extraction (shared backbone)",
         "key-signature-aware parse → diatonic interval classes (step/skip/leap)\nsliding window, repeat collapse → top motif per piece",
         "prep")
    node(0.60, 5.44, 1.75, 1.30, "Labeled real scores",
         "%motif:v1 + %motif:abc\nheader lines\n(217k pieces)",
         "prep", MSO_SHAPE.FOLDED_CORNER)
    node(2.50, 5.44, 1.75, 1.30, "Synthetic crops",
         "±5 bars around each\nmotif occurrence, ×3\n(651k excerpts)",
         "prep", MSO_SHAPE.FOLDED_CORNER)
    arrow(2.42, 2.88, 2.42, 3.16); arrow(2.42, 3.94, 2.42, 4.22)
    arrow(1.90, 5.16, 1.48, 5.44); arrow(2.95, 5.16, 3.38, 5.44)

    # ---------- Panel 2: Training ----------
    panel(4.60, 4.15, "train", "2 · Training")
    node(4.85, 2.10, 3.65, 0.78, "Pretrained NotaGen",
         "hierarchical patch-level encoder + char-level decoder (ABC notation)",
         "train")
    node(4.85, 3.16, 3.65, 0.94, "Phase 1 — synthetic curriculum",
         "fine-tune on motif-labeled crops: association between\n%motif prompt and local content is strong by construction",
         "train")
    node(4.85, 4.38, 3.65, 0.94, "Phase 2 — real scores",
         "full pieces with extracted-motif labels\nattention bias β=+4 added to logits at motif key positions",
         "train")
    node(4.85, 5.60, 3.65, 1.14, "Matched evaluation",
         "each phase eval'd on its own distribution AND bias config\nbest checkpoint selected on real, bias-on eval",
         "train")
    arrow(6.67, 2.88, 6.67, 3.16); arrow(6.67, 4.10, 6.67, 4.38); arrow(6.67, 5.32, 6.67, 5.60)
    # cross-panel: artifacts feed the two phases
    arrow(4.25, 5.85, 4.85, 4.85, color=STAGES["prep"][0])   # crops -> phase 1 region
    arrow(2.42 + 0.9, 6.5, 4.85, 6.1, color=STAGES["prep"][0], dash="dash")  # labeled real -> phase 2/eval

    # ---------- Panel 3: Inference / Generation ----------
    panel(8.85, 4.15, "infer", "3 · Inference / Generation")
    node(9.10, 2.10, 3.65, 0.86, "User motif (creative seed)",
         "(0, 3, -1, -1)\n“leap up, then two steps down” — transposition-invariant",
         "infer", mono_body=False)
    node(9.10, 3.24, 3.65, 0.86, "Prompt assembly",
         "period · composer · instrumentation\n+ %motif:v1:step_skip_leap: 0,3,-1,-1",
         "infer")
    node(9.10, 4.38, 3.65, 0.86, "Biased autoregressive decoding",
         "β on motif key positions keeps the seed salient\ntop-k/top-p sampling, streamed bar patches",
         "infer")
    node(9.10, 5.52, 3.65, 1.22, "Engraved score + audio",
         "full piece containing & developing the motif\nscored by the transposition-invariant,\nenharmonic-aware containment evaluator",
         "infer", MSO_SHAPE.FOLDED_CORNER)
    arrow(10.92, 2.96, 10.92, 3.24); arrow(10.92, 4.10, 10.92, 4.38); arrow(10.92, 5.24, 10.92, 5.52)
    # cross-panel: trained model feeds decoding
    arrow(8.50, 4.85, 9.10, 4.81, color=STAGES["train"][0])

    # ---- Slide 2: data scaling curve ----
    s = title_slide(prs, "Fig 2 — Motif adoption vs. real-data scale",
                    "Total body containment over 11 motifs, n=550 generations/model")
    cd = CategoryChartData()
    cd.categories = [MODEL_LABEL[m] for m in MODELS]
    totals = [sum(body.get((m, mt), 0) for mt in MOTIF_ORDER) / 5.50 for m in MODELS]
    cd.add_series("body containment (%)", totals)
    ch = add_chart(s, XL_CHART_TYPE.COLUMN_CLUSTERED, cd, 1.2, 1.8, 8.0, 5.2, "% of generations")
    ch.plots[0].has_data_labels = True
    ch.plots[0].data_labels.number_format = '0.0"%"'; ch.plots[0].data_labels.number_format_is_linked = False
    tb = s.shapes.add_textbox(Inches(9.6), Inches(2.2), Inches(3.4), Inches(3.5)).text_frame
    tb.word_wrap = True
    tb.text = ("Monotonic: 47.3 → 57.3 → 60.7 → 62.7%.\n\n"
               "Diminishing returns beyond 50k; gains concentrate in mid-distribution motifs "
               "(see Fig 3).")
    for p in tb.paragraphs: p.font.size = Pt(12)

    # ---- Slide 3: frequency dependence ----
    s = title_slide(prs, "Fig 3 — Adoption tracks training-set motif frequency",
                    "Body containment per motif (n=50 each), motifs ordered by frequency rank")
    cd = CategoryChartData()
    cd.categories = MOTIF_ORDER
    for m in ["irish20k_evalfix", "irish50k_r2", "irishfull"]:
        cd.add_series(MODEL_LABEL[m], [2 * body.get((m, mt), 0) for mt in MOTIF_ORDER])
    add_chart(s, XL_CHART_TYPE.COLUMN_CLUSTERED, cd, 0.7, 1.8, 12.0, 5.2, "% of generations")

    # ---- Slide 4: training dynamics ----
    s = title_slide(prs, "Fig 4 — Training dynamics (matched eval)",
                    "Real-phase eval loss per epoch; bias-on eval; best epoch marked in caption")
    cd = CategoryChartData()
    cd.categories = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"]
    cd.add_series("20k (10 real ep.)", (0.1554, 0.1558, 0.1592, 0.1652, 0.1688, 0.1761, 0.1829, 0.1905, 0.1967, 0.2035))
    cd.add_series("50k r2 (3 real ep.)", (0.1592, 0.1592, 0.1600, None, None, None, None, None, None, None))
    cd.add_series("216k (1 real ep.)", (0.1706, None, None, None, None, None, None, None, None, None))
    add_chart(s, XL_CHART_TYPE.LINE_MARKERS, cd, 0.9, 1.8, 8.6, 5.2, "eval loss (real, bias-on)")
    tb = s.shapes.add_textbox(Inches(9.8), Inches(2.2), Inches(3.2), Inches(4.0)).text_frame
    tb.word_wrap = True
    tb.text = ("More real data flattens overfitting: 20k rises +0.048 over 10 epochs; "
               "50k is flat over 3; 216k trains a single token-matched epoch.\n\n"
               "Note: eval sets differ per corpus — curves are not directly comparable "
               "in absolute value; shapes are the point.")
    for p in tb.paragraphs: p.font.size = Pt(11)

    # ---- Slide 5: ablation placeholder ----
    s = title_slide(prs, "Fig 5 — Ablations (PLACEHOLDER: jobs running)",
                    "Full recipe vs bias-off inference vs pretrained+prompt (no recipe) vs unconditioned base rate")
    cd = CategoryChartData()
    cd.categories = ["full recipe", "bias-off", "no recipe", "base rate"]
    cd.add_series("body containment (%) [TBD]", (62.7, None, None, None))
    add_chart(s, XL_CHART_TYPE.COLUMN_CLUSTERED, cd, 1.2, 1.8, 8.0, 5.2, "% of generations")
    tb = s.shapes.add_textbox(Inches(9.6), Inches(2.2), Inches(3.4), Inches(3.0)).text_frame
    tb.word_wrap = True
    tb.text = ("Fill from ablation jobs (abl_base / abl_norecipe_* / abl_biasoff_*) "
               "and the no-curriculum training run; re-run this script after parsing.")
    for p in tb.paragraphs: p.font.size = Pt(12)

    # ---- Slide 6: bias dose-response (historical, to re-verify) ----
    s = title_slide(prs, "Fig 6 — Attention-bias dose–response (RE-VERIFY before use)",
                    "Historical ladder measured with the PRE-FIX checker; re-measure on current models before publishing")
    cd = CategoryChartData()
    cd.categories = ["β=2", "β=4", "β=8"]
    cd.add_series("motif containment (%) [old checker]", (34, 44, 5))
    add_chart(s, XL_CHART_TYPE.COLUMN_CLUSTERED, cd, 1.2, 1.8, 8.0, 5.2, "% of generations")
    tb = s.shapes.add_textbox(Inches(9.6), Inches(2.2), Inches(3.4), Inches(3.0)).text_frame
    tb.word_wrap = True
    tb.text = "β=8 collapses generation quality — the optimum β=4 is used everywhere else."
    for p in tb.paragraphs: p.font.size = Pt(12)

    prs.save(OUT)
    print(f"wrote {OUT} ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
