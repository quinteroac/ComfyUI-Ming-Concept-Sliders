import json
import math
import os
import random
import sys

import torch

import comfy.conds
import comfy.patcher_extension
from comfy_api.latest import ComfyExtension, io

WEB_DIRECTORY = "./web"


BIAS_KEY = "ming_concept_bias"
WEIGHT_KEY = "ming_concept_weight"
RAW_KEY = "ming_concept_raw_weight"
CURVE_KEY = "ming_concept_curve"
# Per text token arrays and their value for tokens that are not a slider concept.
NEUTRAL = {BIAS_KEY: 0.0, WEIGHT_KEY: 1.0, RAW_KEY: 1.0, CURVE_KEY: 0.0}
CURVES_OPTION = "ming_concept_curves"
MAX_CURVE_CONCEPTS = 4
# Ming-Image wraps the prompt in "<role>SYSTEM</role>...<role>HUMAN</role>" ... "<|role_end|><role>ASSISTANT</role><image><imagePatch></image>".
# The direct context holds the per token states up to "</role>": 20 head tokens, the prompt, 6 tail tokens.
TEMPLATE_HEAD, DIRECT_TAIL = 20, 6
MODE_TOOLTIP = "weight_norm scales the concept's attention values and keeps each attention output at the magnitude of the concept as written, weight scales them with no limit, attention scales how much the image attends to the concept."


def extra_conds_with_concept(original, cond, values):
    def extra_conds(**kwargs):
        context = kwargs.get("cross_attn")
        direct = kwargs.get("direct_context")
        if kwargs.get("prompt_type") != "positive" or context is None or direct is None:
            return original(**kwargs)
        # Same place as writing ", concept" at the end of the prompt, before the template closes the user turn.
        # The query tokens stay as they are: the DiT reads the concept from the direct context.
        split = direct.shape[1] - DIRECT_TAIL
        kwargs["direct_context"] = torch.cat((direct[:, :split], cond.to(direct), direct[:, split:]), dim=1)
        # The per text token arrays follow the DiT's text order: the query tokens, then the direct context.
        split += context.shape[1]
        for key, neutral in NEUTRAL.items():
            array = kwargs.get(key)
            value = values.get(key, neutral)
            if array is None and value == neutral:
                continue
            if array is None:
                array = torch.full((1, context.shape[1] + direct.shape[1]), neutral)
            kwargs[key] = torch.cat((array[:, :split], torch.full((array.shape[0], cond.shape[1]), value, dtype=array.dtype, device=array.device), array[:, split:]), dim=1)
        out = original(**kwargs)
        # An earlier node's patch has already extended the arrays it received with its own concept.
        for key in NEUTRAL:
            if key in kwargs and key not in out:
                out[key] = comfy.conds.CONDRegular(kwargs[key])
        return out
    return extra_conds


def with_mask(args, bias):
    # Ming-Image passes its (empty) mask positionally after the head count; chained overrides each add their own logit bias.
    previous = args[1] if len(args) > 1 else None
    mask = bias if previous is None else previous + bias
    return (args[0], mask) + tuple(args[2:])


def concept_attention(previous, bias, weight, raw, image_len):
    def override(func, q, k, v, *args, **kwargs):
        def attention(v, args=args):
            if previous is not None:
                return previous(func, q, k, v, *args, **kwargs)
            return func(q, k, v, *args, **kwargs)
        text_len = next(a for a in (bias, weight, raw) if a is not None).shape[1]
        # The context refiner sees only the text, the joint blocks the text first and then the image, the noise refiner only the image.
        if k.shape[2] not in (text_len, text_len + image_len):
            return attention(v)
        rest = k.shape[2] - text_len
        if bias is not None:
            # log(w) on the logits multiplies the concept's share of attention by w before softmax renormalizes.
            args = with_mask(args, torch.nn.functional.pad(bias.to(q), (0, rest))[:, None, None, :])
        def scale(array):
            return torch.nn.functional.pad(array.to(v), (0, rest), value=1.0)[:, None, :, None]
        if raw is not None:
            # weight: the concept's values scaled as they are, the attention output grows with them.
            v = v * scale(raw)
        if weight is None:
            return attention(v, args)
        out = attention(v * scale(weight), args)
        # The concepts as written set the magnitude, so the weights only turn each head's output.
        # The backend returns the heads concatenated, [b, s, heads * dim_head].
        heads = args[0]
        reference = attention(v, args).unflatten(-1, (heads, -1))
        out = out.unflatten(-1, (heads, -1))
        return (out * (reference.norm(dim=-1, keepdim=True) / out.norm(dim=-1, keepdim=True).clamp(min=1e-6))).flatten(-2)
    return override


def image_tokens(model, x, ref_frames):
    p = model.patch_size
    frames = x.shape[2] if x.ndim == 5 else 1
    return frames * -(-x.shape[-2] // p) * -(-x.shape[-1] // p) + sum(-(-r.shape[-2] // p) * -(-r.shape[-1] // p) for r in ref_frames or [])


def install_override(executor, x, transformer_options, kwargs, bias, weight, raw):
    image_len = image_tokens(executor.class_obj, x, kwargs.get("ref_frames"))
    return {**transformer_options, "optimized_attention_override": concept_attention(transformer_options.get("optimized_attention_override"), bias, weight, raw, image_len)}


def concept_attention_wrapper(executor, x, timestep, context, num_tokens, attention_mask=None, transformer_options={}, **kwargs):
    bias = kwargs.pop(BIAS_KEY, None)
    weight = kwargs.pop(WEIGHT_KEY, None)
    raw = kwargs.pop(RAW_KEY, None)
    if bias is not None or weight is not None or raw is not None:
        transformer_options = install_override(executor, x, transformer_options, kwargs, bias, weight, raw)
    return executor(x, timestep, context, num_tokens, attention_mask, transformer_options=transformer_options, **kwargs)


def curve_value(points, x):
    """Piecewise linear through the points, flat past the ends."""
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else y1
    return points[-1][1]


def curve_step(sigma, sample_sigmas, steps):
    """Where the current step falls on a curve axis of `steps` steps; a sampler with another step count is stretched onto it."""
    sample = sample_sigmas[:-1].tolist()
    i = min(range(len(sample)), key=lambda j: abs(sample[j] - sigma))
    return i * (steps - 1) / max(len(sample) - 1, 1)


def curve_position(axis, steps, sigma_max):
    """x on a curve axis of `steps` points for the current sigma: the sampler step, or the noise level from sigma_max (x = 0) down to 0 (x = steps - 1)."""
    if axis == "noise":
        return lambda sigma, sample_sigmas: min(max(1.0 - sigma / sigma_max, 0.0), 1.0) * (steps - 1)
    return lambda sigma, sample_sigmas: curve_step(sigma, sample_sigmas, steps)


def concept_curve_wrapper(executor, x, timestep, context, num_tokens, attention_mask=None, transformer_options={}, **kwargs):
    numbers = kwargs.pop(CURVE_KEY, None)
    if numbers is not None:
        curves = transformer_options[CURVES_OPTION]
        sigma = transformer_options["sigmas"].max().item()
        strengths = [max(curve_value(points, position(sigma, transformer_options["sample_sigmas"])), 0.0) for points, position, _ in curves]
        index = numbers.long()
        modes = {mode for _, _, mode in curves}

        def table(m):
            # Row 0 is every token that is not a curve concept.
            return torch.tensor([1.0] + [s if mode == m else 1.0 for s, (_, _, mode) in zip(strengths, curves)], device=numbers.device)[index]
        weight = table("weight_norm") if "weight_norm" in modes else None
        raw = table("weight") if "weight" in modes else None
        bias = table("attention").log() if "attention" in modes else None
        transformer_options = install_override(executor, x, transformer_options, kwargs, bias, weight, raw)
    return executor(x, timestep, context, num_tokens, attention_mask, transformer_options=transformer_options, **kwargs)


def mode_input():
    return io.Combo.Input("mode", options=["weight_norm", "weight", "attention"], tooltip=MODE_TOOLTIP)


def encode_concept(clip, concept):
    return clip.encode_from_tokens_scheduled(clip.tokenize(", " + concept), show_pbar=False)[0][1]["direct_context"][:, TEMPLATE_HEAD:-DIRECT_TAIL].clone()


def add_concept(patched, clip, concept, strength, mode):
    if mode == "attention":
        values = {BIAS_KEY: math.log(strength)}
    elif mode == "weight":
        values = {RAW_KEY: strength}
    else:
        values = {WEIGHT_KEY: strength}
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WEIGHT_KEY, concept_attention_wrapper)
    patched.add_object_patch("extra_conds", extra_conds_with_concept(patched.get_model_object("extra_conds"), encode_concept(clip, concept), values))


def apply_concepts(model, clip, weighted, mode):
    patched = model.clone()
    for concept, strength in weighted:
        add_concept(patched, clip, concept, strength, mode)
    return patched


def parse_concept(text, default_max):
    """'fisheye lens#4' -> ('fisheye lens', 4.0)."""
    concept, sep, num = text.rpartition("#")
    if sep and concept.strip():
        try:
            return concept.strip(), float(num)
        except ValueError:
            pass
    return text, default_max


def parse_line(line):
    parts = [t.strip() for t in line.split("|") if t.strip()]
    default_max = 1.5 if len(parts) <= 2 else 1.0
    return [parse_concept(t, default_max) for t in parts]


def line_weights(concepts, value):
    """One concept: strength 0..max. Two: -max_low..max_high, sign picks the concept.
    More: position 0..N, 0 is off, k is concept k at its max, fractions crossfade neighbours."""
    if len(concepts) == 1:
        return [(concepts[0][0], value)]
    if len(concepts) == 2:
        return [(concepts[1][0] if value > 0 else concepts[0][0], abs(value))]
    i = min(math.floor(value), len(concepts))
    t = value - i
    out = []
    if i >= 1:
        out.append((concepts[i - 1][0], (1.0 - t) * concepts[i - 1][1]))
    if i < len(concepts):
        out.append((concepts[i][0], t * concepts[i][1]))
    return out


class MingConceptSlider(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MingConceptSlider",
            display_name="Ming Concept Slider",
            category="Ming/conditioning",
            description="Adds a concept to the end of the positive prompt as a model patch. Negative values use concept_low, positive values use concept_high.",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip", tooltip="The Ming-Image CLIP that encodes the prompts."),
                io.String.Input("concept_low", default=""),
                io.String.Input("concept_high", default=""),
                mode_input(),
                io.Float.Input("strength", default=0.0, min=-1.5, max=1.5, step=0.05, display_mode=io.NumberDisplay.slider),
                io.Boolean.Input("bypass", default=False, tooltip="Returns the model unchanged, to compare against the image without the concepts."),
            ],
            outputs=[io.Model.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, clip, concept_low, concept_high, mode, strength, bypass):
        if bypass:
            return io.NodeOutput(model)
        concept, strength = (concept_high if strength > 0 else concept_low).strip(), abs(strength)
        if strength == 0 or not concept:
            return io.NodeOutput(model)
        return io.NodeOutput(apply_concepts(model, clip, [(concept, strength)], mode))


class MingConceptSliders(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MingConceptSliders",
            display_name="Ming Concept Sliders",
            category="Ming/conditioning",
            description="One slider per line of concepts. A single concept sets its strength, 'low | high' is a bipolar axis, three or more concepts crossfade in order. 'concept#2' sets that concept's max strength.",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip", tooltip="The Ming-Image CLIP that encodes the prompts."),
                io.String.Input("concepts", multiline=True, default="neutral expression | broad smile\nlow angle shot | high angle shot\nfisheye lens\nwatercolor painting"),
                io.String.Input("slider_values", default="{}"),
                mode_input(),
                io.Boolean.Input("bypass", default=False, tooltip="Returns the model unchanged, to compare against the image without the concepts."),
            ],
            outputs=[io.Model.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, clip, concepts, slider_values, mode, bypass):
        if bypass:
            return io.NodeOutput(model)
        values = json.loads(slider_values or "{}")
        weighted = []
        for line in concepts.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            weighted += line_weights(parse_line(line), float(values.get(line, 0.0)))
        # Negative weights or attention have no learned meaning, so a single concept only goes up from 0.
        weighted = [(concept, w) for concept, w in weighted if w > 0]
        if not weighted:
            return io.NodeOutput(model)
        return io.NodeOutput(apply_concepts(model, clip, weighted, mode))


def parse_curves(concepts, curves, steps, max_strength):
    """One curve per concept line: points (step, strength) sorted by step and kept inside the axes. A concept with no curve stays as written."""
    concepts = [c.strip() for c in concepts.splitlines() if c.strip() and not c.strip().startswith("#")]
    if len(concepts) > MAX_CURVE_CONCEPTS:
        raise ValueError(f"Ming Concept Curves takes up to {MAX_CURVE_CONCEPTS} concepts, one per line; got {len(concepts)}.")
    return [(concept, sorted((min(max(float(x), 0.0), steps - 1.0), min(max(float(y), 0.0), max_strength)) for x, y in curves.get(concept) or [[0, 1.0]]))
            for concept in dict.fromkeys(concepts)]


class MingConceptCurves(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MingConceptCurves",
            display_name="Ming Concept Curves",
            category="Ming/conditioning",
            description="Up to 4 concepts, one per line, each with its own strength curve over the sampling steps. Draw the curves in the editor: x is the step or the noise level (axis), y the strength (1 = the concept as written).",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip", tooltip="The Ming-Image CLIP that encodes the prompts."),
                io.String.Input("concepts", multiline=True, default="fisheye lens\nbroad smile"),
                io.Int.Input("steps", default=12, min=1, max=10000, tooltip="Points on the curve axis. With axis steps they are the sampler steps, and a sampler with another step count is stretched onto the curve; with axis noise they set how finely the noise axis is divided."),
                io.Float.Input("max_strength", default=1.5, min=0.05, max=sys.float_info.max, step=0.05, tooltip="Top of the strength axis."),
                io.String.Input("curves", default="{}"),
                mode_input(),
                io.Boolean.Input("bypass", default=False, tooltip="Returns the model unchanged, to compare against the image without the concepts."),
                io.Combo.Input("axis", options=["steps", "noise"], default="steps", tooltip="What x is: the sampler step, or the noise level from full noise (left) to none (right), so a curve does the same at any schedule or step count."),
            ],
            outputs=[io.Model.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, clip, concepts, steps, max_strength, curves, mode, bypass, axis):
        curves = parse_curves(concepts, json.loads(curves or "{}"), steps, max_strength)
        if bypass or not curves:
            return io.NodeOutput(model)
        patched = model.clone()
        # The concepts go in as written; each step scales them in attention.
        options = patched.model_options.setdefault("transformer_options", {})
        registered = list(options.get(CURVES_OPTION, []))
        position = curve_position(axis, steps, float(patched.get_model_object("model_sampling").sigma_max))
        for concept, points in curves:
            registered.append((points, position, mode))
            patched.add_object_patch("extra_conds", extra_conds_with_concept(patched.get_model_object("extra_conds"), encode_concept(clip, concept), {CURVE_KEY: float(len(registered))}))
        options[CURVES_OPTION] = registered
        patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, CURVE_KEY, concept_curve_wrapper)
        return io.NodeOutput(patched)


with open(os.path.join(os.path.dirname(__file__), "styles.json"), encoding="utf-8") as f:
    STYLES = json.load(f)


def parse_style_codes(code, mix):
    """code 123 and mix '456::0.5, 789' -> [(123, 1.0), (456, 0.5), (789, 1.0)], like --sref 123 456::0.5 789."""
    codes = [(code, 1.0)]
    for part in mix.replace(",", " ").split():
        number, _, weight = part.partition("::")
        if not number.isdigit():
            raise ValueError("Ming Style Code: mix takes style codes with an optional ::weight, like 456::0.5, got '%s'." % part)
        codes.append((int(number), float(weight) if weight else 1.0))
    return codes


def code_styles(code, count):
    """The styles a code stands for and their weights, which average 1 (a flat Dirichlet)."""
    rng = random.Random(code)
    picked = rng.sample(STYLES, min(count, len(STYLES)))
    draws = [rng.expovariate(1.0) for _ in picked]
    return [(style, count * d / sum(draws)) for style, d in zip(picked, draws)]


def mix_styles(codes, count):
    weights = {}
    for code, code_weight in codes:
        for style, w in code_styles(code, count):
            weights[style["id"]] = (style, weights.get(style["id"], (style, 0.0))[1] + w * code_weight)
    return [entry for entry in weights.values() if entry[1] > 0]


class MingStyleCode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MingStyleCode",
            display_name="Ming Style Code",
            category="Ming/conditioning",
            description="A style from a number, like Midjourney's --sref codes: each code picks a few styles from the pack's style list with random weights and adds them to the positive prompt. Randomize the code to look for styles and keep the codes you like.",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Int.Input("code", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True, tooltip="The style code. The same code gives the same styles and weights on any prompt."),
                io.String.Input("mix", default="", tooltip="More codes to blend in, each with an optional ::weight (code is weight 1), like 456::0.5, 789."),
                io.Int.Input("styles_per_code", default=2, min=1, max=8, tooltip="How many styles each code mixes."),
                io.Float.Input("strength", default=1.5, min=0.0, max=4.0, step=0.05, display_mode=io.NumberDisplay.slider, tooltip="How much the image attends to the styles, like --sw; 1 is as written. 0 is the model unchanged."),
                io.Boolean.Input("bypass", default=False, tooltip="Returns the model unchanged, to compare against the image without the style."),
            ],
            outputs=[io.Model.Output(), io.String.Output(display_name="styles", tooltip="The styles the codes picked, with their weights.")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, clip, code, mix, styles_per_code, strength, bypass):
        styles = mix_styles(parse_style_codes(code, mix), styles_per_code)
        listing = "\n".join("%s (%s) %.2f" % (style["label"], style["category"], w) for style, w in styles)
        if bypass or strength == 0:
            return io.NodeOutput(model, listing)
        return io.NodeOutput(apply_concepts(model, clip, [(style["text"], w * strength) for style, w in styles], "attention"), listing)


class MingConceptSlidersExtension(ComfyExtension):
    async def get_node_list(self):
        return [MingConceptSlider, MingConceptSliders, MingConceptCurves, MingStyleCode]


async def comfy_entrypoint():
    return MingConceptSlidersExtension()
