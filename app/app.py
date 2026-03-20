import os
import gradio as gr
from rembg import remove

TEMPLATES = {
    "Пустой лист": "assets/template.png",
    "Персонаж 1": "assets/img01_template.png",
    "Персонаж 2": "assets/img02_template.png",
    "Персонаж 3": "assets/img03_template.png",
    "Персонаж 4": "assets/img04_template.png",
    "Персонаж 5": "assets/img05_template.png",
}

ANIMATIONS = {
    "Персонаж 5": "assets/img05_anim.gif",
}


def on_select(name):
    path = TEMPLATES[name]
    return path, path


def remove_bg(image):
    return remove(image)


def animate(name):
    path = ANIMATIONS.get(name)
    if path:
        return gr.update(value=path, visible=True), gr.update(visible=False)
    return gr.update(value=None, visible=False), gr.update(value="Анимация пока доступна только для Персонажа 5.", visible=True)


css = ".gradio-container { max-width: 640px !important; margin: auto !important; }"

with gr.Blocks(title="Ink-to-Motion", theme=gr.themes.Soft(primary_hue="orange"), css=css) as demo:
    gr.Markdown("# Ink-to-Motion")

    gr.Markdown("### Шаг 1. Скачай шаблон")
    gr.Markdown("Выбери шаблон, распечатай на A4.")
    selector = gr.Dropdown(choices=list(TEMPLATES.keys()), value="Пустой лист", show_label=False)
    preview = gr.Image(value="assets/template.png", show_label=False, interactive=False, height=300)
    download = gr.DownloadButton("Скачать шаблон", value="assets/template.png", variant="secondary")
    selector.change(fn=on_select, inputs=selector, outputs=[preview, download])

    gr.Markdown("### Шаг 2. Сфотографируй рисунок")
    gr.Markdown("Нарисуй персонажа и покажи что получилось.")
    image_in = gr.Image(show_label=False, type="pil")

    gr.Markdown("### Шаг 3. Оцифровка")
    btn = gr.Button("Оцифровать", variant="primary")
    image_out = gr.Image(label="Результат")
    btn.click(fn=remove_bg, inputs=image_in, outputs=image_out)

    gr.Markdown("### Шаг 4. Анимация")
    gr.Markdown("Посмотри как персонаж оживает!")
    anim_btn = gr.Button("Анимировать", variant="primary")
    anim_out = gr.Image(label="Анимация", visible=False)
    anim_msg = gr.Markdown(visible=False)
    anim_btn.click(fn=animate, inputs=selector, outputs=[anim_out, anim_msg])

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
