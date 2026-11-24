# ComfyUI Ming Concept Sliders

Concept sliders, concept curves and style codes for Ming-Image 0.1 Design in ComfyUI. Each node adds short concepts (`broad smile`, `fisheye lens`) to the positive prompt as a `MODEL` patch, so you can fade them in or scale them up, and chain nodes like LoRAs. Port of the [MOSS](https://github.com/quinteroac/ComfyUI-MOSS-Concept-Sliders) and [MiniMax H3 Image](https://github.com/quinteroac/ComfyUI-MiniMax-H3-Concept-Sliders) packs.

## How it works

Ming-Image's text encoder gives the DiT two kinds of text tokens: 256 query tokens, and the per token states of the prompt (the direct context). The concept is encoded on its own as `, concept` and its direct context tokens go in at the end of the prompt, before the chat template closes the user turn, where they would land if you wrote `, concept` at the end of the prompt. The query tokens are left as they are. A strength of 1 is the concept as written there; in tests it gives nearly the same image as writing it in the prompt.

It works the same with Text Encode Ming Image Edit and reference images, so a slider can change an edit (a smile on an outfit change, for example).

## Install

Clone the repo into `ComfyUI/custom_nodes` and restart ComfyUI. The nodes need no extra Python packages.

## Setup

Use the usual Ming-Image files: the Design model in `models/diffusion_models`, the Ling text encoder in `models/text_encoders` (CLIP type `qwen_image`) and the Ming-Image VAE.

1. **Load Diffusion Model** → **Ming Concept Sliders** (`model`) → ModelSamplingFlux → BasicGuider.
2. **Load CLIP** → **Ming Concept Sliders** (`clip`). Use the same CLIP that encodes your prompts.
3. Leave the slider concepts out of the prompt, because the node adds them. Set a strength on the node's sliders and queue.

The nodes are under `Ming/conditioning`. `example_workflows` has API workflows for each node (Euler, 12 steps, CFG 1, 1024×1024).

## Nodes

**Ming Concept Slider**: one bipolar slider. Negative strength uses `concept_low` and positive strength uses `concept_high`.

**Ming Concept Sliders**: one slider per line of `concepts`.

- `fisheye lens`: a single concept, strength 0 to 1.5.
- `neutral expression | broad smile`: a bipolar axis, −1.5 to 1.5.
- `overcast | golden hour | night`: three or more concepts crossfade in order. 0 is off, and k is concept k at full strength.
- `fisheye lens#4`: sets that concept's maximum strength.
- Lines starting with `#` are skipped.

**Ming Concept Curves**: up to 4 concepts, one per line, each with its own strength curve over the sampling steps, drawn in the node's editor (x is the step or, with `axis` noise, the noise level; y the strength). Click a concept in the legend to edit its curve, click to add a point, drag to move it, double-click to delete it, right-click for preset shapes. `steps` defaults to 12, Ming-Image's usual step count.

**Ming Style Code**: a style from a number, like Midjourney's `--sref` codes. Each `code` picks `styles_per_code` styles (default 2) from `styles.json` with random weights that average 1 and adds their descriptions to the prompt in attention mode. The same code gives the same mix on any prompt; the `styles` output lists what it picked. `mix` blends more codes, like `456::0.5, 789`. `strength` works like `--sw` (default 1.5). Ming-Image follows the descriptions closely, including the objects a style names: Kinetic Art brings its rods and discs. `styles.json` holds 645 styles: names and categories from [comfyui-stylebook](https://github.com/EnragedAntelope/comfyui-stylebook) (MIT, see `STYLES_LICENSE`), each description rewritten as a short list of visual traits, the same list as the MOSS and H3 packs.

The slider and curve nodes have a `bypass` switch that returns the model unchanged, to compare against the image without the concepts.

## Modes

| Mode | What it does |
|---|---|
| attention | Adds log(w) to the attention logits on the concept's tokens in the context refiner and every joint block. |
| weight_norm | Scales the attention values of the concept's tokens and keeps each head's attention output at the magnitude it has with the concept as written. Runs attention twice. |

Negative strengths have no meaning in these modes, so a single concept goes from 0 up. A concept such as `fisheye lens` saturates early: strength 1 to 8 gives about the same lens.

The modes run attention through an override, so the attention backend chosen with Model Attention Backend is replaced by ComfyUI's default one while a concept is active.

## Limits

The concept is encoded on its own. The Ling text encoder is causal, so the prompt tokens and the 256 query tokens never saw the concept, unlike writing it in the prompt text. Only the positive prompt gets the concepts, inserted before the default template tail.

## Tests

From the ComfyUI root, with its Python environment:

```bash
python custom_nodes/ComfyUI-Ming-Concept-Sliders/tests/test_sliders.py
```

They build a tiny random Ming-Image model on the CPU and check that strength 1 matches the concept written in the prompt (with and without edit references), that strengths change the image, that chained nodes match one node, and that flat curves match the sliders.

## License

MIT. `styles.json` is derived from comfyui-stylebook (MIT), see `STYLES_LICENSE`.
