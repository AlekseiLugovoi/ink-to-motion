import os
import gradio as gr
from fastapi import Request
from fastapi.responses import JSONResponse

try:
    from .config import CHARS, DEFAULT_CHAR, ASSETS_DIR, BACKGROUND_VIDEO_PATH
    from .pipeline import (
        decode_camera_photo, detect_aruco, align,
        do_color_transfer, do_animation, do_composite,
        auto_process,
        _on_auto_char_change, _on_step_char_change, _render_qr_html,
        _render_preview_with_skeleton, _generate_animation_preview,
    )
    from .camera import CAMERA_HEAD, make_camera_panel, CSS
except ImportError:
    from config import CHARS, DEFAULT_CHAR, ASSETS_DIR, BACKGROUND_VIDEO_PATH
    from pipeline import (
        decode_camera_photo, detect_aruco, align,
        do_color_transfer, do_animation, do_composite,
        auto_process,
        _on_auto_char_change, _on_step_char_change, _render_qr_html,
        _render_preview_with_skeleton, _generate_animation_preview,
    )
    from camera import CAMERA_HEAD, make_camera_panel, CSS

# ---------------------------------------------------------------------------
#  Initial values for default character
# ---------------------------------------------------------------------------

_default_char = CHARS.get(DEFAULT_CHAR) if DEFAULT_CHAR else None
_default_template = _default_char.get("template") if _default_char else None
_default_preview = _render_preview_with_skeleton(_default_char) if _default_char else None
_default_anim = _generate_animation_preview(_default_char) if _default_char else None

# ---------------------------------------------------------------------------
#  Gradio UI
# ---------------------------------------------------------------------------

char_choices = list(CHARS.keys())

with gr.Blocks(title="Ink-to-Motion") as demo:
    gr.Markdown("# Ink-to-Motion")
    with gr.Tabs():

        # === Авто ===
        with gr.Tab("Авто"):
            with gr.Column(elem_id="main-wrap"):
                gr.Markdown("Выбери персонажа, отсканируй QR или загрузи фото.")

                with gr.Accordion("Шаг 1. Шаблон и QR", open=True):
                    auto_char_selector = gr.Dropdown(
                        choices=char_choices,
                        value=DEFAULT_CHAR,
                        label="Персонаж",
                        show_label=False,
                    )
                    auto_template_preview = gr.Image(
                        value=_default_template,
                        label="Шаблон",
                        interactive=False,
                        height=220,
                        elem_classes=["preset-media"],
                    )
                    auto_download_btn = gr.DownloadButton(
                        "Скачать шаблон",
                        value=_default_template,
                        variant="secondary",
                    )
                    auto_qr_container = gr.HTML(
                        _render_qr_html(DEFAULT_CHAR),
                        elem_classes=["qr-center-block"],
                    )

                auto_char_id = gr.Textbox(
                    value=DEFAULT_CHAR or "", elem_id="auto-char-id", container=False,
                )

                gr.HTML(make_camera_panel("camera-data-auto"))
                camera_data_auto = gr.Textbox(elem_id="camera-data-auto", container=False)
                image_in_auto = gr.Image(show_label=False, type="pil", sources=["upload"])
                camera_data_auto.change(
                    fn=decode_camera_photo, inputs=camera_data_auto, outputs=image_in_auto,
                )

                gr.Markdown("Анимация")
                auto_gif = gr.Video(show_label=False, autoplay=True, loop=True)
                gr.Markdown("Наложение на фон")
                auto_composite = gr.HTML(
                    '<div style="min-height:200px;display:flex;align-items:center;'
                    'justify-content:center;color:#9ca3af;border:1px dashed #d1d5db;'
                    'border-radius:12px;background:#f8fafc;">'
                    '<span style="font-size:40px;opacity:0.4;">&#127916;</span></div>'
                )

                auto_char_selector.change(
                    fn=_on_auto_char_change,
                    inputs=auto_char_selector,
                    outputs=[auto_char_id, auto_template_preview, auto_download_btn, auto_qr_container],
                )

                image_in_auto.change(
                    fn=auto_process, inputs=[image_in_auto, auto_char_id],
                    outputs=[auto_gif, auto_composite],
                )

        # === Пошагово ===
        with gr.Tab("Пошагово"):
            with gr.Column(elem_id="main-wrap"):

                with gr.Accordion("Шаг 0. Выбери персонажа", open=True):
                    char_selector = gr.Dropdown(
                        choices=char_choices,
                        value=DEFAULT_CHAR,
                        label="Персонаж",
                        show_label=False,
                    )
                    with gr.Accordion("Превью", open=False):
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=1, min_width=0):
                                preset_preview = gr.Image(
                                    value=_default_preview,
                                    label="Скелет",
                                    interactive=False,
                                    height=250,
                                    elem_classes=["preset-media"],
                                )
                            with gr.Column(scale=1, min_width=0):
                                preset_anim = gr.Video(
                                    value=_default_anim,
                                    label="Анимация",
                                    autoplay=True,
                                    loop=True,
                                    height=250,
                                )

                with gr.Accordion("Шаг 1. Шаблон и QR", open=True):
                    gr.Markdown("Распечатай шаблон на A4, раскрась и сфотографируй.")
                    template_preview = gr.Image(
                        value=_default_template,
                        label="Шаблон",
                        interactive=False,
                        height=220,
                        elem_classes=["preset-media"],
                    )
                    download_btn = gr.DownloadButton(
                        "Скачать шаблон",
                        value=_default_template,
                        variant="secondary",
                    )
                    qr_container = gr.HTML(
                        _render_qr_html(DEFAULT_CHAR),
                        elem_classes=["qr-center-block"],
                    )

                char_selector.change(
                    fn=_on_step_char_change,
                    inputs=char_selector,
                    outputs=[
                        preset_preview, preset_anim,
                        template_preview, download_btn, qr_container,
                    ],
                )

                with gr.Accordion("Шаг 2. Сфотографируй рисунок", open=False):
                    gr.Markdown("Покажи, что получилось.")
                    gr.HTML(make_camera_panel("camera-data"))
                    camera_data = gr.Textbox(elem_id="camera-data", container=False)
                    image_in = gr.Image(show_label=False, type="pil", sources=["upload"])
                    camera_data.change(
                        fn=decode_camera_photo, inputs=camera_data, outputs=image_in,
                    )

                with gr.Accordion("Шаг 3. Выравнивание", open=False):
                    align_btn = gr.Button("Выровнять", variant="primary")
                    aligned_out = gr.Image(show_label=False, interactive=False, height=300)
                    align_btn.click(fn=align, inputs=image_in, outputs=aligned_out)

                with gr.Accordion("Шаг 4. Перенос цвета", open=False):
                    gr.Markdown("Цвета с фото переносятся на оригинального персонажа по маске.")
                    ct_btn = gr.Button("Перенести цвет", variant="primary")
                    color_out = gr.Image(label="Результат", interactive=False, height=300)
                    ct_btn.click(
                        fn=do_color_transfer,
                        inputs=[aligned_out, char_selector],
                        outputs=color_out,
                    )

                frames_state = gr.State([])

                with gr.Accordion("Шаг 5. Анимация", open=False):
                    anim_btn = gr.Button("Анимировать", variant="primary")
                    anim_out = gr.Video(show_label=False, autoplay=True, loop=True)
                    anim_btn.click(
                        fn=do_animation,
                        inputs=[aligned_out, char_selector],
                        outputs=[anim_out, frames_state],
                    )

                with gr.Accordion("Шаг 6. На фоне", open=False):
                    gr.Markdown("Персонаж попадает на подводный фон.")
                    composite_btn = gr.Button("Поместить на фон", variant="primary")
                    composite_out = gr.HTML()
                    composite_file = gr.File(label="Скачать HTML")
                    composite_btn.click(
                        fn=do_composite,
                        inputs=[frames_state, char_selector],
                        outputs=[composite_out, composite_file],
                    )

    # --- Read ?char= from URL on load (fixes QR routing to wrong character) ---
    def _on_url_load(request: gr.Request):
        char_id = request.query_params.get("char", DEFAULT_CHAR)
        if char_id not in CHARS:
            char_id = DEFAULT_CHAR
        char = CHARS[char_id]
        template = char.get("template")
        return char_id, char_id, template, template, _render_qr_html(char_id)

    demo.load(
        fn=_on_url_load,
        outputs=[auto_char_selector, auto_char_id,
                 auto_template_preview, auto_download_btn, auto_qr_container],
    )

# ---------------------------------------------------------------------------
#  FastAPI endpoint for ArUco detection
# ---------------------------------------------------------------------------

@demo.app.post("/api/detect_aruco")
async def _detect_aruco_api(request: Request):
    data = await request.json()
    count = detect_aruco(data.get("frame", ""))
    return JSONResponse({"count": int(count)})

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860)),
    theme=gr.themes.Soft(primary_hue="orange"),
    css=CSS,
    head=CAMERA_HEAD,
    allowed_paths=[str(ASSETS_DIR.resolve()), os.path.dirname(BACKGROUND_VIDEO_PATH)],
)
