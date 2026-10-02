"""MotiGen: interactive motif-conditioned music generation on Hugging Face Spaces."""
import os
# spaces must be imported before torch for ZeroGPU's CUDA emulation.
try:
    import spaces
except ImportError:
    spaces = None

from pathlib import Path
from html import escape
import secrets
import threading

import gradio as gr

from engine import generate, load_model
from motifs import DEFAULT_NOTES, build_prompt
from outputs import OUTPUT_ROOT, export_result
from retries import MAX_ATTEMPTS, generate_verified

ROOT = Path(__file__).resolve().parent
STYLES = {" · ".join(fields): fields
          for fields in (tuple(line.strip().split("_"))
                         for line in (ROOT / "assets/prompts.txt").read_text().splitlines())
          if len(fields) == 3 and fields[2] == "Art Song"}
DEFAULT_STYLE = next(key for key in STYLES if "Schubert" in key and "Art Song" in key)
MODEL = None
MODEL_LOCK = threading.Lock()
if os.environ.get("SPACE_ID"):
    MODEL = load_model()


def stream_model(prompt, temperature, seed, bars, mode, pattern, notes, tempo, rhythm=None):
    global MODEL
    with MODEL_LOCK:
        if MODEL is None:
            MODEL = load_model()
        yield from generate_verified(lambda **kwargs: generate(MODEL, **kwargs), prompt,
                                     temperature, seed, bars, mode, pattern, notes, tempo, rhythm=rhythm)


if spaces is not None:
    stream_model = spaces.GPU(duration=120)(stream_model)


def run_generation(mode, abstract, notes, style, bars, tempo, temperature, seed, occurrences, rhythm=""):
    if style not in STYLES:
        raise gr.Error("Select one of the available Art Song styles.")
    try:
        prompt, pattern, snippet, rhythm = build_prompt(mode, abstract, notes, STYLES[style], occurrences, rhythm)
        if int(bars) not in (8, 16, 32) or not 50 <= int(tempo) <= 180 or not 0.6 <= float(temperature) <= 1.5:
            raise ValueError("Use the available length, tempo, and creativity controls.")
        seed = int(seed)
        if not -1 <= seed <= 2**32 - 1:
            raise ValueError("Seed must be -1 (random) or an integer from 0 to 4294967295.")
    except (ValueError, TypeError, OverflowError) as error:
        raise gr.Error(str(error)) from None
    seed = secrets.randbelow(2**32) if seed == -1 else seed
    motif_label = f"Motif: {pattern}" + (f" · Rhythm: {rhythm}" if rhythm else "")
    prompt_details = ("\n\n<details><summary>Model prompt used</summary><pre>"
                      + escape(prompt) + "</pre></details>")
    def generation_details(summary):
        return ("<details><summary>Generation details</summary><p>" + escape(summary)
                + "</p>" + prompt_details + "</details>")
    # Keep progress brief; diagnostic information is collapsed by default.
    yield "Composing…", prompt, "", "", None, None, None, "", generation_details(f"Seed: {seed} · {motif_label}")
    final = None
    try:
        # Concrete input verifies exact pitches and durations already; the rhythm line only
        # tightens verification for abstract motifs.
        verify_rhythm = rhythm if mode == "Abstract motif" else None
        for update in stream_model(prompt, float(temperature), seed, int(bars), mode, pattern, notes, int(tempo), verify_rhythm):
            final = update
            yield ("Composing…", update.text,
                   gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                   generation_details(f"{update.message} · Seed: {update.seed} · {motif_label}"))
        if final is None:
            raise ValueError("Generation returned no music. Please try again.")
        if final.stage != 'verified':
            # Keep the result panels empty when every candidate fails verification.
            yield "No motif match found. Try again.", gr.skip(), "", "", None, None, None, "", gr.skip()
            return
        yield "Preparing score…", gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
        abc, xml, abc_file, xml_file, audio, score, warning = export_result(final.text, int(tempo), motif_match=final.match)
        status = warning or ""
        measures = ', '.join(map(str, final.match.bars))
        details = generation_details(f"{final.message} · Verified {final.match.kind} · Measure(s): {measures} · {final.eligible} eligible · Seed: {final.seed} · {motif_label}")
        yield status, final.text, abc, xml, abc_file, xml_file, audio, score, details
    except Exception as error:
        # Keep already streamed text available to diagnose an incomplete generation.
        raise gr.Error(str(error)) from error


CSS = """
.gradio-container {width:100%!important;max-width:1380px!important;margin:auto;background:#f5f0e6!important;color:#302a24;}
.gradio-container main {padding:28px 32px 40px!important;}
#hero {position:relative;display:flex;align-items:center;gap:22px;padding:12px 0 28px;margin-bottom:18px;border-bottom:3px double #bba786;}
#hero::after {content:"";position:absolute;width:7px;height:7px;bottom:-5px;left:50%;transform:rotate(45deg);background:#9b7845;box-shadow:0 0 0 5px #f5f0e6;}
#hero .hero-mark {width:56px;height:70px;flex:none;color:#9b7845;}
#hero h1 {font:400 64px/1.05 Georgia,"Times New Roman",serif;letter-spacing:-2.6px;color:#302a24;margin:0 0 10px;}
#hero p {font-size:14px;line-height:1.6;letter-spacing:.15px;color:#716353;margin:0;}
#workspace {gap:30px!important;}
#workspace h3 {font:400 25px/1.25 Georgia,"Times New Roman",serif;color:#43392f;margin:6px 0 8px;}
#score-panel {padding:0 0 0 24px;border-left:1px solid #ded3c1;}
#generate {background:#302a24;border:1px solid #302a24;color:#fffdf7;min-height:48px;letter-spacing:.35px;font-weight:500;box-shadow:0 2px 0 #b69b70;}
#generate:hover {background:#4a3e31;border-color:#4a3e31;}
#status {padding:10px 14px;background:#eee5d5;border:1px solid #ded0b7;border-left:3px solid #a18150;border-radius:3px;color:#594a37;font-size:13px;}
#status:not(:has(p, pre, ul, ol)) {display:none;}
#player {border:1px solid #ded3c1;background:#fffcf5;}
#score-tabs>.tab-nav {border-bottom:1px solid #d6c8b1;gap:8px;}
#score-tabs>.tab-nav button {font-size:13px;color:#786a59;padding:12px 8px;}
#score-tabs>.tab-nav button.selected {color:#3b3023;border-color:#94713d;font-weight:600;}
#score-tabs>.tabitem {padding:16px 0 0;border:0;background:transparent;}
#score-preview {background:#fffdf7;border:1px solid #dfd3c0;border-radius:3px;box-shadow:0 3px 12px #5c452c08;}
.score-empty {height:430px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:22px;padding:24px;box-sizing:border-box;}
.score-empty svg {width:190px;max-width:80%;color:#b5a487;}
.score-empty p {font:italic 19px Georgia,"Times New Roman",serif;color:#88785f;text-align:center;}
#downloads button,#downloads a {font-size:13px;min-height:43px;border:1px solid #cbbba2;background:transparent;color:#51412e;box-shadow:none;}
#downloads button:hover,#downloads a:hover {background:#ede3d1;}
#result-details {font-size:12px;color:#756650;}
#result-details details {margin-top:8px;}
#result-details summary {cursor:pointer;}
#result-details pre {background:#ece4d7;border:1px solid #dbceb8;border-radius:3px;padding:14px;color:#4c4134;}
.note-hint {font-size:12px;color:#756650;}
.gradio-container button:focus-visible,.gradio-container input:focus-visible,.gradio-container a:focus-visible {outline:2px solid #9a7138;outline-offset:3px;}
footer {display:none!important;}
@media (max-width:700px) {
  .gradio-container,.gradio-container .block {min-width:0!important;}
  .gradio-container main {padding:18px 16px 28px!important;min-width:0!important;}
  .gradio-container .row {flex-direction:column!important;}
  .gradio-container .column {min-width:0!important;width:100%!important;}
  #hero {gap:14px;padding:8px 0 22px;margin-bottom:8px;}
  #hero .hero-mark {width:38px;height:52px;}
  #hero h1 {font-size:46px;letter-spacing:-1.7px;}
  #hero p {font-size:12px;}
  #workspace {gap:20px!important;}
  #score-panel {border-left:0;border-top:1px solid #d6c8b1;padding:18px 0 0;}
  #workspace h3 {font-size:23px;}
  .score-empty {height:320px;}
}
"""


def classical_theme():
    colors = dict(
        body_background_fill="#f5f0e6", body_text_color="#302a24",
        body_text_color_subdued="#746653", background_fill_primary="#fffdf7",
        background_fill_secondary="#eee6d8", border_color_primary="#d8ccb9",
        border_color_accent="#a18150", border_color_accent_subdued="#dfd0b7",
        color_accent_soft="#eee2cc", block_background_fill="#f9f5ed",
        block_border_color="#ded3c1", block_label_background_fill="transparent",
        block_label_text_color="#6d5b43", block_title_background_fill="transparent",
        block_title_text_color="#6d5b43", input_background_fill="#fffdf7",
        input_border_color="#d4c5ad", input_border_color_focus="#9b7845",
        input_placeholder_color="#998972",
        button_primary_background_fill="#302a24", button_primary_background_fill_hover="#4a3e31",
        button_primary_text_color="#fffdf7", button_secondary_background_fill="#faf6ee",
        button_secondary_background_fill_hover="#eee4d3", button_secondary_text_color="#51412e",
        button_secondary_border_color="#d2c2a8", checkbox_label_background_fill="#faf6ee",
        checkbox_label_background_fill_selected="#eee2cc", checkbox_label_text_color="#66543c",
        checkbox_label_text_color_selected="#3c2d1c", checkbox_label_border_color="#d2c2a8",
        checkbox_label_border_color_selected="#aa8752", checkbox_background_color_selected="#92713e",
        checkbox_border_color_selected="#92713e", checkbox_border_color="#bba98e",
        checkbox_background_color="#fffdf7", slider_color="#a18150",
        code_background_fill="#f4ecdf",
    )
    return gr.themes.Base(primary_hue="stone", secondary_hue="amber", neutral_hue="stone",
                          radius_size="sm", font=["system-ui", "Arial", "sans-serif"],
                          font_mono=["ui-monospace", "monospace"]).set(
        **colors, **{name + "_dark": value for name, value in colors.items()},
        color_accent="#94713d", block_border_width="1px",
        block_label_text_size="12px", block_label_text_weight="500",
        block_title_text_size="12px", block_title_text_weight="500",
        button_primary_shadow="none", button_secondary_shadow="none",
        input_shadow="none", input_shadow_focus="0 0 0 1px #a18150",
    )


def build_demo():
    with gr.Blocks(title="MotiGen · From motif to music",
                   css=CSS + (ROOT / "assets/motif-editor.css").read_text(),
                   js="async () => {\n" + (ROOT / "assets/vexflow-bravura-5.0.0.js").read_text()
                      + "\nawait document.fonts.ready;\n("
                      + (ROOT / "assets/motif-editor.js").read_text() + ")();\n}",
                   theme=classical_theme(),
                   delete_cache=(3600, 3600)) as demo:
        gr.HTML('''<header id="hero"><svg class="hero-mark" viewBox="0 0 56 70" aria-hidden="true"><path d="M1 1H55V69H1Z M5 5H51V65H5Z" fill="none" stroke="currentColor" stroke-width=".65"/><path d="M10 27H46 M10 32H46 M10 37H46 M10 42H46 M10 47H46" stroke="currentColor" stroke-width=".5"/><ellipse cx="25" cy="43" rx="5" ry="3.2" transform="rotate(-22 25 43)" fill="currentColor"/><path d="M29.5 42V19L39 16V35" stroke="currentColor" stroke-width="1.4" fill="none"/><ellipse cx="34.5" cy="36" rx="5" ry="3.2" transform="rotate(-22 34.5 36)" fill="currentColor"/></svg><div><h1>MotiGen</h1><p>Create a motif. Generate, play, and export your music.</p></div></header>''')
        with gr.Row(equal_height=False, elem_id="workspace"):
            with gr.Column(scale=4, min_width=340, elem_id="compose-panel"):
                gr.Markdown("### Your motif")
                # Keep the API contract; browser inputs are serialized directly
                # from the visual editor when Generate is clicked.
                mode = gr.Radio(["Abstract motif", "Concrete notes"], value="Abstract motif", visible=False)
                abstract = gr.Textbox(value="0,3,-1,-1", visible=False)
                notes = gr.Dataframe(value=DEFAULT_NOTES, headers=["Pitch", "Beats"],
                                     datatype=["str", "str"], type="array", visible=False)
                gr.HTML((ROOT / "assets/motif-editor.html").read_text())
                rhythm = gr.Textbox(value="", label="Rhythm · optional, abstract motif only",
                                    placeholder="e.g. 1,1/2,1/2,2 — one duration per note, in beats",
                                    max_lines=1, elem_id="rhythm")
                gr.Markdown("Durations are ratios to the beat (the meter's bottom number: under 4/4 “1” is a quarter, under 6/8 an eighth). Leave blank to let the model choose the rhythm. Concrete notes carry their own durations.", elem_classes="note-hint")
                gr.Markdown("### Composition")
                style = gr.Dropdown(list(STYLES), value=DEFAULT_STYLE, label="Art Song style")
                with gr.Row():
                    bars = gr.Radio([8, 16, 32], value=16, label="Measures (up to)")
                    tempo = gr.Slider(50, 180, value=100, step=1, label="Playback tempo · BPM")
                with gr.Accordion("Generation settings", open=False):
                    temperature = gr.Slider(0.6, 1.5, value=1.2, step=0.05, label="Creativity")
                    seed = gr.Number(-1, precision=0, label="Seed · −1 for a new result")
                    occurrences = gr.Slider(1, 8, value=3, step=1, label="Requested motif occurrences · concrete input")
                    gr.Markdown(f"Generates {MAX_ATTEMPTS} candidates and favors the most verified motif occurrences, then the most distinct pitch–duration pairs. Ties on both favor the earlier candidate. The requested occurrence count guides the model; one match is required. Concrete input uses 4/4.", elem_classes="note-hint")
                with gr.Row():
                    start = gr.Button("Generate music", variant="primary", elem_id="generate")
                    stop = gr.Button("Stop", variant="secondary")
            with gr.Column(scale=6, min_width=400, elem_id="score-panel"):
                gr.Markdown("### Your score")
                status = gr.Markdown("", elem_id="status")
                player = gr.Audio(label="Playback", type="filepath", interactive=False, elem_id="player")
                with gr.Tabs(elem_id="score-tabs"):
                    with gr.Tab("Sheet music"):
                        score = gr.HTML('''<div class="score-empty"><svg viewBox="0 0 190 55" aria-hidden="true"><path d="M0 12H190 M0 20H190 M0 28H190 M0 36H190 M0 44H190" fill="none" stroke="currentColor" stroke-width=".5"/><path d="M60 12V44 M130 12V44" stroke="currentColor" stroke-width=".5"/><path d="M84 23H106V27H84Z" fill="currentColor"/></svg><p>Your score will appear here.</p></div>''', elem_id="score-preview")
                    with gr.Tab("ABC"):
                        abc = gr.Code(label="ABC notation", language=None, interactive=False, lines=20)
                    with gr.Tab("MusicXML"):
                        xml = gr.Code(label="MusicXML source", language="html", interactive=False, lines=20)
                    with gr.Tab("Live generation"):
                        live = gr.Textbox(label="Model output · updates while composing", interactive=False, lines=20)
                with gr.Row(elem_id="downloads"):
                    abc_download = gr.DownloadButton("Download ABC")
                    xml_download = gr.DownloadButton("Download MusicXML (.xml)")
                details = gr.Markdown("", elem_id="result-details")
        event = start.click(run_generation, [mode, abstract, notes, style, bars, tempo, temperature, seed, occurrences, rhythm],
                            [status, live, abc, xml, abc_download, xml_download, player, score, details],
                            api_name="generate", concurrency_limit=1, concurrency_id="model",
                            js="""(...args) => {
                              if (!window.motigenEditor) throw new Error('The motif editor is loading. Please try again.');
                              const input = window.motigenEditor.getInput();
                              args[0] = input.mode;
                              args[1] = input.abstract;
                              args[2] = {headers: ['Pitch', 'Beats'], data: input.notes, metadata: null};
                              return args;
                            }""")
        stop.click(lambda: "Stopped.", outputs=status, cancels=[event], queue=False)
    return demo.queue(max_size=12, default_concurrency_limit=1)


if __name__ == "__main__":
    build_demo().launch(server_name=os.environ.get("SERVER_NAME", "0.0.0.0"), server_port=int(os.environ.get("PORT", 7860)),
                        allowed_paths=[str(OUTPUT_ROOT)], show_error=True)
