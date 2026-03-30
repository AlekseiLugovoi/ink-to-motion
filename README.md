# ink-to-motion

Рисунок на бумаге → оцифровка → анимация \
DEMO: https://ink-to-motion-production.up.railway.app

**TODO:**

<details>
<summary>2026-03-22</summary>

- [x] Сделать нормальный перенос цветов
    - Маска на внутреннюю часть шаблона?
- [ ] Сделать больше персонажей

</details>

<details>
<summary>2026-03-18</summary>

- [x] Новый шаблон
- [x] Новые целевые персонажи
- [x] Перенос цвета на детерминированного персонажа
- [x] Попытка анимации (скелет + триангуляция)
- [x] Наложение анимации на фоне
- [x] MVP сервиса через UI (gradio)
- [x] Поднять сервис на Railway

### Заметки (поштормили)

- предпосылки:
    - есть один заранее известный (полностью детерменированный) персонаж
        - заранее хорошо прорисован в нормальной позе
        - наложен скелет (?)
        - понятная анимация (раскадровка или все-таки этот скелет как-то двигать)
- приземлить вырезание персонажа (как?)
- квадратики мб показывают еще и направление
- прототип ручной отрисовки ок, но не для масштабирования
- ок если генерация создается только один раз (не каждый раз при запросе анимации)
- мэш 3д - разметить скелет (руками)

</details>

<details>
<summary>2026-03-14</summary>

- [x] Причесать гит: Общий пайплайн, примеры, TODO
- [ ] 10-20шт примеров (для флоу B)
- [ ] Пример раскадровки
- [x] 🆕 Выравнивание по ArUco-маркерам шаблон
- [x] 🆕 Примеры выравнивания
- [x] 🆕 Проработка флоу A: четкое наложение красок

</details>



## Пайплайн

### 0. Подготовка шаблона

Печатаем лист с ArUco-маркерами по углам:
- внутри — пустой шаблон или контур персонажа
- маркеры позволяют точно выровнять фото на следующем шаге
- служат ориентиром при съёмке — все 4 маркера должны быть в кадре
- помогают оценить качество фото — сильное искажение маркеров → предупреждение «переснимите»

| Пустой | С персонажем |
|---|---|
| <img src="preprocessing/output/template.png" width="300"> | <img src="preprocessing/output/img01_template.png" width="300"> |

### 1. Выравнивание

> Фото шаблона → детекция маркеров → гомография → выравнивание на координаты шаблона.
> Качество выравнивания оцениваем по отклонению углов маркеров (px) после гомографии.

| Фото | Выровненное |
|---|---|
| <img src="preprocessing/output/img01_photo.jpg" width="300"> | <img src="preprocessing/output/img01_photo_aligned.jpg" width="300"> |
| <img src="preprocessing/output/img01_photo_bad.jpg" width="300"> | <img src="preprocessing/output/img01_photo_bad_aligned.jpg" width="300"> |
| <img src="preprocessing/output/img03_photo.jpg" width="300"> | <img src="preprocessing/output/img03_photo_aligned.jpg" width="300"> |
| <img src="preprocessing/output/img03_photo_bad.jpg" width="300"> | <img src="preprocessing/output/img03_photo_bad_aligned.jpg" width="300"> |
| <img src="preprocessing/output/img04_photo.jpg" width="300"> | <img src="preprocessing/output/img04_photo_aligned.jpg" width="300"> |
| <img src="preprocessing/output/img04_photo_bad.jpg" width="300"> | <img src="preprocessing/output/img04_photo_bad_aligned.jpg" width="300"> |

**TODO:**
- [ ] более точная оценка качества фото (дисторсия, сдвиг в центре, а не только на маркерах)
- [ ] детекция бликов/теней → попытка автокоррекции или подсказка «перефоткайте»
- [ ] замер времени на весь шаг (детекция маркеров + гомография + warp)

### 2. Оцифровка персонажа

Два сценария в зависимости от того, был ли контур на шаблоне:

**Флоу A — известный контур** (шаблон с персонажем):
> Контур уже есть в шаблоне → после выравнивания фото точно совпадает с оригиналом → детерминированно переносим цвет с фото на чистый контур.

| Оригинальный контур | Выровненное фото | Результат |
|---|---|---|
| <img src="preprocessing/input/img01_raw.jpg" width="250"> | <img src="preprocessing/output/img01_photo_aligned.jpg" width="250"> | <img src="preprocessing/output/img01_photo_aligned_digitized.png" width="250"> |

**TODO:**
- [ ] убрать артефакты переноса (шум фото проходит через saturation-порог)
- [ ] хромакей для переноса цвета — на шаблоне задать зону (маску) внутри персонажа, откуда строго забирать цвета из разукрашки вместо пороговой фильтрации по saturation
- [ ] варианты заливки: попиксельно с фото vs усреднённый цвет по области
- [ ] замер времени

**Флоу B — произвольный рисунок** (пустой шаблон):
> Контура заранее нет → нужно отсегментировать персонажа из фото, убрать фон, маркеры и мусор → RGBA.

| Кроп (drawing_bbox) | Сегментация (rembg) |
|---|---|
| <img src="preprocessing/output/img04_photo_aligned.jpg" width="300"> | <img src="preprocessing/output/img04_photo_aligned_digitized.png" width="300"> |

**TODO:**
- [ ] сравнить модели сегментации (rembg, SAM2, и др.)
- [ ] замер времени

### 3. Улучшение рисунка (опционально)

> Набросок может быть бледным, низкого разрешения или слишком простым для хорошей анимации. Можно улучшить до подачи на следующие шаги.

**Апскейл** — повышение разрешения:
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) — x4 апскейл, хорошо работает на рисунках/аниме
- [SwinIR](https://github.com/JingyunLiang/SwinIR) — transformer-based super resolution

**Пример апскейла** (Real-ESRGAN x4, реализация в [`preprocess.ipynb`](rnd/preprocess.ipynb)):

| До (483x511) | После (1932x2044) |
|---|---|
| <img src="rnd/output/img04_clean.png" width="200"> | <img src="rnd/output/img04_upscaled.png" width="200"> |

**Стилизация** — из наброска в "продакшн" картинку (img2img):
- [ControlNet](https://github.com/lllyasviel/ControlNet) + SD — линии/скетч как контроль, стиль через промпт
- [T2I-Adapter](https://github.com/TencentARC/T2I-Adapter) — аналогичный подход, легче

**Раскраска** — автоматическая раскраска ч/б рисунка:
- [Style2Paints](https://github.com/lllyasviel/style2paints) — AI-раскраска скетчей
- ControlNet lineart + цветовой промпт

**Нормализация позы** — привести персонажа в T-pose/A-pose для удобства rigging и анимации:
- [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) + ControlNet openpose — сохраняем внешность персонажа, задаём целевую позу скелетом
- [CharacterGen](https://github.com/zjp-shadow/CharacterGen) — генерация character sheet (фронт, бок, спина) из одного ракурса

**Пример нормализации позы** (реализация в [`normalize_pose.ipynb`](rnd/normalize_pose.ipynb)):

| Оригинал | Qwen-Image-Edit (локально) | Gemini 3.1 Flash (OpenRouter) |
|---|---|---|
| <img src="rnd/output/img04_clean.png" width="150"> | <img src="rnd/output/img04_tpose_qwen.png" width="150"> | <img src="rnd/output/img04_tpose_gemini.png" width="150"> |

### 4. Pose Estimation (опционально)

> Скелет нужен для управляемой анимации — чтобы персонаж двигался по заданному паттерну (танец, ходьба), а не "как модель решит". Без скелета можно обойтись (см. 5.2 без pose), но с ним результат предсказуемее.

**Автоматически** — задача специфическая, готовых решений для рисунков нет. Нужно:
- Классифицировать тип скелета (двуногий, четвероногий, водный и т.д.)
- Под каждый тип — своя модель pose estimation
- Путь: разметка данных + дообучение (YOLO-pose или аналоги)

Ссылки:
- [YOLOv8-pose](https://docs.ultralytics.com/tasks/pose/) — лёгкая модель, можно дообучить на свои keypoints
- [DWPose](https://github.com/IDEA-Research/DWPose) — SOTA pose estimation, используется в MusePose
- [RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) (MMPose) — быстрый, есть animal-модели

**Вручную** — пользователь сам размечает ключевые точки скелета по шаблону.

**Пример ручной разметки** (реализация в [`pose.ipynb`](rnd/pose.ipynb)):

| Персонаж | Скелет |
|---|---|
| <img src="rnd/output/img03_clean.png" width="200"> | <img src="rnd/output/img03_pose.png" width="200"> |

### 5. Оживление

#### 5.1 Bone-анимация (без AI)

> Классический подход: разрезаем персонажа на части по скелету и вращаем вокруг суставов. Предсказуемо, быстро, работает на любом устройстве. Движения можно брать из mocap-данных (BVH-файлы).
>
> **Ограничение:** хорошо работает только на stick-figure (палочные рисунки) и контурных персонажах с тонкими конечностями. На объёмных рисунках с заливкой части разъезжаются, видны швы и артефакты — для таких лучше использовать генеративный i2v (п. 5.2).

- Требует pose estimation (шаг 4)
- Движения из mocap баз: [CMU MoCap](http://mocap.cs.cmu.edu/), [Mixamo](https://www.mixamo.com/)

**Пример** (реализация в [`animate_bone.ipynb`](rnd/animate_bone.ipynb)):

| Stick-figure (работает) | Заливка (артефакты) |
|---|---|
| <img src="rnd/output/img01_animated.gif" width="200"> | <img src="rnd/output/img03_animated.gif" width="200"> |

#### 5.2 Mesh-деформация (без AI)

> Деформируем картинку целиком по скелету: Delaunay-триангуляция по ключевым точкам → движение точек по паттерну (синусоида) → per-triangle affine warp. Работает на CPU, без нейросетей.

- Требует pose estimation (шаг 4)
- Паттерны движения задаются вручную (качание хвоста, плавников и т.д.)

**Пример** (реализация в [`prepare_template.ipynb`](preprocessing/prepare_template.ipynb)):

| Персонаж | Скелет | Триангуляция | Анимация |
|---|---|---|---|
| <img src="preprocessing/input/img05.png" width="200"> | <img src="preprocessing/output/img05_skeleton.png" width="200"> | <img src="preprocessing/output/img05_triangulation.png" width="200"> | <img src="preprocessing/output/img05_anim.gif" width="200"> |

#### 5.3 Image-to-Video

> Генеративная модель оживляет рисунок целиком — самый универсальный подход, работает на любых персонажах.

**С pose estimation** — управляемая анимация через видео-референс:
- [MusePose](https://github.com/TMElyralab/MusePose) — pose-driven i2v, танец по референсу (16GB VRAM @ 512x512)
- [MagicAnimate](https://github.com/magic-research/magic-animate) — аналогичный подход от ByteDance
- [Animate Anyone](https://github.com/HumanAIGC/AnimateAnyone) — Alibaba, есть community-репродукции

**Без pose estimation** — генеративная модель сама решает как двигать:
- [Wan 2.1](https://github.com/Wan-Video/Wan2.1) — i2v от Alibaba, лёгкие и тяжёлые версии
- [CogVideoX](https://github.com/THUDM/CogVideo) — i2v от Tsinghua
- [Stable Video Diffusion](https://github.com/Stability-AI/generative-models) — i2v от Stability AI

**Пример** (реализация в [`i2v.ipynb`](rnd/i2v.ipynb)):

| Оригинал | Wan 2.1 VACE 1.3B (локально) | Kling v2.5 Turbo Pro (fal.ai) |
|---|---|---|
| <img src="rnd/output/img03_clean.png" width="200"> | <img src="rnd/output/img03_i2v.gif" width="200"> | <img src="rnd/output/img03_i2v_kling.gif" width="200"> |
| | [mp4](rnd/output/img03_i2v.mp4) | [mp4](rnd/output/img03_i2v_kling.mp4) |

#### 5.4 Image-to-3D

> Перевод рисунка в 3D-модель — можно крутить, анимировать в движке, использовать в играх.

- [TripoSR](https://github.com/VAST-AI-Research/TripoSR) — быстрый, ~6GB VRAM, базовый меш
- [Hunyuan3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) — Tencent, PBR текстуры, 10-29GB VRAM
- [Trellis-2](https://github.com/Microsoft/TRELLIS) — Microsoft, лучшее качество, 16GB+ VRAM

**Пример** (реализация в [`i2mesh.ipynb`](rnd/i2mesh.ipynb)).
Для раскраски меша по оригинальному рисунку можно использовать Hunyuan3D-Paint (21GB+ VRAM).

| Hunyuan3D-2 shape only (локально) | Trellis (fal.ai) |
|---|---|
| <img src="rnd/output/img03_mesh.gif" width="200"> | <img src="rnd/output/img03_mesh_trellis.gif" width="200"> |
| [glb](rnd/output/img03_mesh.glb) | [glb](rnd/output/img03_mesh_trellis.glb) |

## Инфраструктура

Каждый шаг протестирован в двух режимах — бесплатно на локальном GPU и через облачные API:

| | Локально (open-source) | Облако (SOTA) |
|---|---|---|
| **GPU** | RTX 4070 Ti SUPER 16GB | — |
| **Нормализация позы** | Qwen-Image-Edit | Gemini 3.1 Flash ([OpenRouter](https://openrouter.ai/)) |
| **Image-to-Video** | Wan 2.1 VACE 1.3B | Kling v2.5 Turbo Pro ([fal.ai](https://fal.ai/), ~$0.35/5с) |
| **Image-to-3D** | Hunyuan3D-2 | Trellis ([fal.ai](https://fal.ai/), ~$0.02) |

## Похожие проекты

- [AnimatedDrawings](https://github.com/facebookresearch/AnimatedDrawings) (Meta) — полный пайплайн от рисунка до анимации
- [Lester](https://github.com/rtous/lester-code) — анимация детских рисунков
- [Monster Mash](https://github.com/google/monster-mash) (Google) — скетч → 3D модель → анимация в браузере

## Выводы

- **Проприетарные модели через API** ([OpenRouter](https://openrouter.ai/), [fal.ai](https://fal.ai/)) дают SOTA-качество без нагрузки на локальное железо: неограниченная параллельность, нет проблем с VRAM, всегда свежие модели, pay-per-use дешевле содержания GPU при малых объёмах.
- **Предобработка — самый важный шаг.** Чем чище вход (персонаж на белом фоне, без мусора, нормализованная поза), тем лучше результат на всех следующих этапах. Мусор на входе → мусор на выходе.
- **Самый быстрый путь к результату:** предобработка (очистка фона + нормализация позы) → image-to-video с хорошим промптом. Минимум шагов, максимум эффекта.
