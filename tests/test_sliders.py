import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.samplers
import comfy.supported_models

spec = importlib.util.spec_from_file_location("ming_sliders", Path(__file__).resolve().parents[1] / "__init__.py")
sliders = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sliders)

HEAD, TAIL = sliders.TEMPLATE_HEAD, sliders.DIRECT_TAIL
QUERIES, CAP_DIM, DIM = 4, 16, 256
SIGMAS = torch.tensor([1.0, 0.6, 0.3, 0.0])


class FakeClip:
    """Encodes ', concept' as a few query tokens and a direct context of the template head, one token per word and the template tail, like the Ming-Image text encoder."""
    def __init__(self):
        self.generator = torch.Generator().manual_seed(7)
        self.encoded = {}

    def tokenize(self, text):
        return text

    def encode_from_tokens_scheduled(self, text, show_pbar=True):
        if text not in self.encoded:
            self.encoded[text] = (torch.randn((1, QUERIES, CAP_DIM), generator=self.generator), torch.randn((1, HEAD + len(text.split()) + TAIL, DIM), generator=self.generator))
        queries, direct = self.encoded[text]
        return [[queries, {"direct_context": direct}]]

    def concept(self, concept):
        self.encode_from_tokens_scheduled(", " + concept)
        return self.encoded[", " + concept][1][:, HEAD:-TAIL]


def conditioning(generator, words, extra=None):
    return [[torch.randn((1, QUERIES, CAP_DIM), generator=generator), {"direct_context": torch.randn((1, HEAD + words + TAIL, DIM), generator=generator), **(extra or {})}]]


class TinyMing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = comfy.supported_models.MingImage(dict(image_model="ming_image", dim=DIM, cap_feat_dim=CAP_DIM, n_layers=2, n_refiner_layers=1, n_heads=2, n_kv_heads=2, axes_dims=[32, 48, 48], axes_lens=[1536, 512, 512], rope_theta=256.0, ffn_dim_multiplier=8.0 / 3.0, z_image_modulation=True, time_scale=1000.0, qk_norm=True, in_channels=16, masked_pad_multiple=32, dtype=torch.float32))
        config.custom_operations = comfy.ops.manual_cast
        model = comfy.model_base.MingImage(config, device=torch.device("cpu"))
        generator = torch.Generator().manual_seed(42)
        state = {name: torch.ones_like(value) if "norm" in name else torch.randn(value.shape, generator=generator) * 0.1
                 for name, value in model.state_dict().items() if name.startswith("diffusion_model.")}
        model.load_state_dict(state, strict=False)
        cls.model = comfy.model_patcher.ModelPatcher(model, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
        cls.prompt = conditioning(generator, 6)
        cls.negative = conditioning(generator, 1)
        cls.edit = conditioning(generator, 6, {"reference_latents": [torch.randn((1, 16, 1, 4, 4), generator=generator)]})
        cls.noise = torch.randn((1, 16, 1, 4, 4), generator=generator)

    @classmethod
    def tearDownClass(cls):
        comfy.model_management.unload_all_models()

    def setUp(self):
        self.clip = FakeClip()

    def sample(self, model, positive=None):
        guider = comfy.samplers.CFGGuider(model)
        guider.set_conds(self.prompt if positive is None else positive, self.negative)
        guider.set_cfg(3.0)
        result = guider.sample(self.noise.clone(), torch.zeros_like(self.noise), comfy.samplers.ksampler("euler"), SIGMAS, disable_pbar=True, seed=42)
        self.assertTrue(torch.isfinite(result).all())
        return result

    def written(self, *concepts, positive=None):
        queries, extra = (self.prompt if positive is None else positive)[0]
        direct = extra["direct_context"]
        direct = torch.cat([direct[:, :-TAIL]] + [self.clip.concept(c) for c in concepts] + [direct[:, -TAIL:]], dim=1)
        return [[queries, {**extra, "direct_context": direct}]]

    def slider(self, concepts, slider_values, mode, model=None):
        return sliders.MingConceptSliders.execute(self.model if model is None else model, self.clip, concepts, slider_values, mode, False)[0]



class SliderTests(TinyMing):
    def test_zero_and_bypass_return_the_model(self):
        self.assertIs(self.slider("broad smile", "{}", "attention"), self.model)
        self.assertIs(sliders.MingConceptSlider.execute(self.model, self.clip, "", "broad smile", "attention", 1.0, True)[0], self.model)
        self.assertIs(sliders.MingConceptSlider.execute(self.model, self.clip, "", "broad smile", "attention", -1.0, False)[0], self.model)

    def test_one_is_the_concept_written_at_the_end(self):
        for mode in ("attention", "weight_norm", "weight"):
            with self.subTest(mode=mode):
                self.setUp()
                patched = self.slider("broad smile", '{"broad smile": 1}', mode)
                torch.testing.assert_close(self.sample(patched), self.sample(self.model, self.written("broad smile")), rtol=0, atol=1e-5)

    def test_strength_changes_the_image(self):
        written = self.sample(self.model, self.written("broad smile"))
        for mode in ("attention", "weight_norm", "weight"):
            with self.subTest(mode=mode):
                self.setUp()
                self.assertGreater((self.sample(self.slider("broad smile", '{"broad smile": 3}', mode)) - written).abs().max().item(), 1e-3)

    def test_edit_references_keep_the_concept_in_place(self):
        for mode in ("attention", "weight_norm", "weight"):
            with self.subTest(mode=mode):
                self.setUp()
                one = self.sample(self.slider("broad smile", '{"broad smile": 1}', mode), self.edit)
                written = self.sample(self.model, self.written("broad smile", positive=self.edit))
                torch.testing.assert_close(one, written, rtol=0, atol=1e-5)
                self.assertGreater((self.sample(self.slider("broad smile", '{"broad smile": 3}', mode), self.edit) - written).abs().max().item(), 1e-3)

    def test_chained_nodes_match_one_node(self):
        values = '{"fisheye lens": 2, "broad smile": 3}'
        chained = self.slider("broad smile", '{"broad smile": 3}', "attention", self.slider("fisheye lens", '{"fisheye lens": 2}', "attention"))
        torch.testing.assert_close(self.sample(chained), self.sample(self.slider("fisheye lens\nbroad smile", values, "attention")), rtol=0, atol=1e-5)

    def test_override_scales_only_the_text(self):
        generator = torch.Generator().manual_seed(3)
        image_len, text_len, heads = 4, 5, 2
        weight = torch.tensor([[1.0, 1.0, 3.0, 3.0, 1.0]])
        bias = weight.log()

        def func(q, k, v, heads, mask=None, **kwargs):
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            return out.transpose(1, 2).flatten(2)

        for n in (text_len + image_len, text_len, image_len):
            q, k, v = (torch.randn((1, heads, n, 8), generator=generator) for _ in range(3))
            plain = func(q, k, v, heads, None)
            if n == image_len:
                torch.testing.assert_close(sliders.concept_attention(None, bias, None, None, image_len)(func, q, k, v, heads, None), plain)
                continue
            logits = torch.zeros((1, 1, 1, n))
            logits[..., 2:4] = math.log(3)
            torch.testing.assert_close(sliders.concept_attention(None, bias, None, None, image_len)(func, q, k, v, heads, None), func(q, k, v, heads, logits))
            scaled = v.clone()
            scaled[:, :, 2:4] *= 3
            normed = sliders.concept_attention(None, None, weight, None, image_len)(func, q, k, v, heads, None).unflatten(-1, (heads, -1))
            expected = func(q, k, scaled, heads, None).unflatten(-1, (heads, -1))
            torch.testing.assert_close(normed.norm(dim=-1), plain.unflatten(-1, (heads, -1)).norm(dim=-1))
            torch.testing.assert_close(normed / normed.norm(dim=-1, keepdim=True), expected / expected.norm(dim=-1, keepdim=True))
            torch.testing.assert_close(sliders.concept_attention(None, None, None, weight, image_len)(func, q, k, v, heads, None), func(q, k, scaled, heads, None))


class CurveTests(TinyMing):
    def curves(self, curves, mode, axis="steps"):
        return sliders.MingConceptCurves.execute(self.model, self.clip, "\n".join(curves), 3, 4.0, json.dumps(curves), mode, False, axis)[0]

    def test_flat_curve_is_the_slider(self):
        for mode in ("attention", "weight_norm", "weight"):
            with self.subTest(mode=mode):
                self.setUp()
                curve = self.sample(self.curves({"fisheye lens": [[0, 2.5]]}, mode))
                torch.testing.assert_close(curve, self.sample(self.slider("fisheye lens", '{"fisheye lens": 2.5}', mode)), rtol=0, atol=1e-5)

    def test_curve_changes_over_the_steps(self):
        flat = self.sample(self.curves({"fisheye lens": [[0, 1.0]]}, "attention"))
        ramp = self.sample(self.curves({"fisheye lens": [[0, 1.0], [2, 4.0]]}, "attention"))
        self.assertGreater((ramp - flat).abs().max().item(), 1e-3)


class StyleCodeTests(TinyMing):
    def test_codes_pick_repeatable_styles(self):
        self.assertEqual(sliders.code_styles(7, 2), sliders.code_styles(7, 2))
        self.assertNotEqual(sliders.code_styles(7, 2), sliders.code_styles(8, 2))
        self.assertAlmostEqual(sum(w for _, w in sliders.code_styles(7, 3)), 3.0, places=6)
        self.assertEqual(len(sliders.mix_styles(sliders.parse_style_codes(7, "8::0"), 2)), 2)

    def test_code_changes_the_image(self):
        model, listing = sliders.MingStyleCode.execute(self.model, self.clip, 7, "", 2, 1.5, False)
        self.assertEqual(len(listing.splitlines()), 2)
        self.assertGreater((self.sample(model) - self.sample(self.model)).abs().max().item(), 1e-3)
        self.assertIs(sliders.MingStyleCode.execute(self.model, self.clip, 7, "", 2, 0, False)[0], self.model)


class ParseTests(unittest.TestCase):
    def test_line_weights(self):
        self.assertEqual(sliders.parse_line("fisheye lens#4"), [("fisheye lens", 4.0)])
        self.assertEqual(sliders.line_weights(sliders.parse_line("a | b"), -0.5), [("a", 0.5)])
        self.assertEqual(sliders.line_weights(sliders.parse_line("a | b | c"), 1.25), [("a", 0.75), ("b", 0.25)])


if __name__ == "__main__":
    unittest.main()
