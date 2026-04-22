import os
import tempfile
import gradio as gr
from fastapi import Request
from fastapi.responses import JSONResponse

from config import CHARS, DEFAULT_CHAR, ASSETS_DIR, BACKGROUND_VIDEO_PATH
from pipeline import (
    decode_camera_photo, detect_aruco, align,
    do_color_transfer, do_animation, do_composite,
    auto_process, add_to_aquarium, clear_aquarium,
    _on_auto_char_change, _on_step_char_change, _render_qr_html,
    _render_preview_with_skeleton, _generate_animation_preview,
    AQUARIUM_EMPTY,
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
    aquarium_state = gr.State([])
    last_fish_state = gr.State(None)
    with gr.Tabs():

        # === Авто ===
        with gr.Tab("Авто"):
            with gr.Column(elem_id="main-wrap"):
                gr.Markdown("Распечатай шаблон, раскрась и сфотографируй.")

                with gr.Accordion("Шаг 1. Распечатай шаблон", open=True):
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

                with gr.Accordion("Шаг 2. Ручной режим (старые шаблоны)", open=False):
                    gr.Markdown("Для старых шаблонов с QR на каждого персонажа. Выбери того же персонажа, что на распечатке.")
                    manual_char_selector = gr.Dropdown(
                        choices=char_choices,
                        value=DEFAULT_CHAR,
                        label="Персонаж",
                        show_label=False,
                    )
                    with gr.Row():
                        with gr.Column(scale=1, min_width=0):
                            manual_qr = gr.HTML(
                                _render_qr_html(DEFAULT_CHAR),
                                elem_classes=["qr-center-block"],
                            )
                        with gr.Column(scale=1, min_width=0):
                            gr.HTML(make_camera_panel("camera-data-manual"))
                            camera_data_manual = gr.Textbox(elem_id="camera-data-manual", container=False)
                            image_in_manual = gr.Image(
                                type="pil", sources=["upload"],
                                height=140, show_label=False,
                            )
                            camera_data_manual.change(
                                fn=decode_camera_photo, inputs=camera_data_manual, outputs=image_in_manual,
                            )

                with gr.Accordion("Шаг 3. Автоопределение (новые шаблоны)", open=True):
                    gr.Markdown("Универсальный QR — персонаж определится по маркерам на фото.")
                    with gr.Row():
                        with gr.Column(scale=1, min_width=0):
                            gr.HTML(
                                _render_qr_html(),
                                elem_classes=["qr-center-block"],
                            )
                        with gr.Column(scale=1, min_width=0):
                            gr.HTML(make_camera_panel("camera-data-auto"))
                            camera_data_auto = gr.Textbox(elem_id="camera-data-auto", container=False)
                            image_in_auto = gr.Image(
                                type="pil", sources=["upload"],
                                height=140, show_label=False,
                            )
                            camera_data_auto.change(
                                fn=decode_camera_photo, inputs=camera_data_auto, outputs=image_in_auto,
                            )

                auto_detected = gr.Markdown()

                with gr.Accordion("Шаг 4. Анимация", open=True):
                    auto_gif = gr.HTML()

                with gr.Accordion("Шаг 5. На фоне", open=True):
                    auto_composite = gr.HTML()
                    add_aquarium_btn = gr.Button(
                        "Добавить в аквариум", variant="secondary",
                    )

                auto_char_selector.change(
                    fn=_on_auto_char_change,
                    inputs=auto_char_selector,
                    outputs=[auto_template_preview, auto_download_btn],
                )

                manual_char_selector.change(
                    fn=lambda cid: _render_qr_html(cid),
                    inputs=manual_char_selector,
                    outputs=manual_qr,
                )

                def _process_photo(photo, selected_char=None):
                    anim_html, composite_html, fish_data, char_id = auto_process(photo, selected_char)
                    if not char_id:
                        info = ""
                    elif selected_char:
                        info = f"Используется персонаж: **{char_id}** (ручной выбор)"
                    else:
                        info = f"Определён персонаж: **{char_id}**"
                    return info, anim_html, composite_html, fish_data

                image_in_auto.change(
                    fn=lambda p: _process_photo(p),
                    inputs=[image_in_auto],
                    outputs=[auto_detected, auto_gif, auto_composite, last_fish_state],
                )
                image_in_manual.change(
                    fn=_process_photo,
                    inputs=[image_in_manual, manual_char_selector],
                    outputs=[auto_detected, auto_gif, auto_composite, last_fish_state],
                )

        # === Аквариум ===
        with gr.Tab("Аквариум"):
            with gr.Column(elem_id="main-wrap"):
                gr.Markdown("Здесь живут твои персонажи. Обработай фото во вкладке **Авто** и нажми **Добавить в аквариум**.")
                aquarium_counter = gr.Markdown("Персонажей: 0 / 5")
                aquarium_display = gr.HTML(AQUARIUM_EMPTY)
                fullscreen_url = gr.Textbox(visible=False, elem_id="fs-url")
                with gr.Row():
                    fullscreen_btn = gr.Button("Полный экран", variant="secondary")
                    clear_aquarium_btn = gr.Button("Очистить аквариум", variant="secondary")
                fullscreen_btn.click(
                    fn=None, inputs=fullscreen_url, outputs=None,
                    js="(url) => { if(url) window.open(url, '_blank'); }",
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

                step_char_state = gr.State("")

                with gr.Accordion("Шаг 3. Выравнивание", open=False):
                    align_btn = gr.Button("Выровнять", variant="primary")
                    aligned_out = gr.Image(show_label=False, interactive=False, height=300)

                    def _step_align(image):
                        aligned, char_id = align(image)
                        return aligned, char_id

                    align_btn.click(
                        fn=_step_align, inputs=image_in,
                        outputs=[aligned_out, step_char_state],
                    )

                with gr.Accordion("Шаг 4. Перенос цвета", open=False):
                    gr.Markdown("Цвета с фото переносятся на оригинального персонажа по маске.")
                    ct_btn = gr.Button("Перенести цвет", variant="primary")
                    color_out = gr.Image(label="Результат", interactive=False, height=300)
                    ct_btn.click(
                        fn=do_color_transfer,
                        inputs=[aligned_out, step_char_state],
                        outputs=color_out,
                    )

                frames_state = gr.State([])

                with gr.Accordion("Шаг 5. Анимация", open=False):
                    anim_btn = gr.Button("Анимировать", variant="primary")
                    anim_out = gr.Video(show_label=False, autoplay=True, loop=True)
                    anim_btn.click(
                        fn=do_animation,
                        inputs=[aligned_out, step_char_state],
                        outputs=[anim_out, frames_state],
                    )

                with gr.Accordion("Шаг 6. На фоне", open=False):
                    gr.Markdown("Персонаж попадает на подводный фон.")
                    composite_btn = gr.Button("Поместить на фон", variant="primary")
                    composite_out = gr.HTML()
                    composite_file = gr.File(label="Скачать HTML")
                    composite_btn.click(
                        fn=do_composite,
                        inputs=[frames_state, step_char_state],
                        outputs=[composite_out, composite_file],
                    )

    def _add_fish(last_fish, state):
        state, display, btn = add_to_aquarium(last_fish, state)
        gr.Info(f"Персонаж добавлен! ({len(state)}/5)")
        return state, display, btn, f"Персонажей: {len(state)} / 5"

    def _clear_fish():
        state, display, btn = clear_aquarium()
        return state, display, btn, "Персонажей: 0 / 5"

    add_aquarium_btn.click(
        fn=_add_fish,
        inputs=[last_fish_state, aquarium_state],
        outputs=[aquarium_state, aquarium_display, fullscreen_url, aquarium_counter],
    )
    clear_aquarium_btn.click(
        fn=_clear_fish,
        inputs=[],
        outputs=[aquarium_state, aquarium_display, fullscreen_url, aquarium_counter],
    )

    # --- Read ?char= from URL on load (pre-selects template for download) ---
    def _on_url_load(request: gr.Request):
        char_id = request.query_params.get("char")
        if char_id and char_id in CHARS:
            return char_id
        return gr.update()

    demo.load(
        fn=_on_url_load,
        outputs=[auto_char_selector],
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
    allowed_paths=[str(ASSETS_DIR.resolve()), os.path.dirname(BACKGROUND_VIDEO_PATH),
                   tempfile.gettempdir()],
)
