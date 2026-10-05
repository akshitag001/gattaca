# Image brief for GATTACA

Two categories, and the distinction matters technically.

| Category | Can be AI-generated / stock? | Why |
|---|---|---|
| **A. Decorative** (hero, section art) | ✅ Yes | Never analysed. Pure visual. |
| **B. Analysed** (pipeline input, stage frames, distance ladder, before/after) | ❌ **No — must be real photographs** | A diffusion model paints ridge-*like* texture, not a real friction-ridge field with coherent orientation and frequency. Measuring PPRP, clarity or minutiae on a generated hand produces numbers about an artifact. It would also repeat the exact flaw in the news coverage we are rebutting (their "proof" image is labelled *AI 생성 이미지*). |

---

## A. Decorative — prompts for the image developer

Deliver as **PNG or high-quality JPEG, portrait 3:4, at least 1600×2133**, no text/watermark/logo baked in.

### A1 — Hero (primary)

> Photorealistic editorial portrait, young adult holding one hand up toward the camera in a casual peace / V sign, hand occupying the foreground and in sharp focus, face softly out of focus behind, shot on 85mm lens at f/2.0, shallow depth of field, moody low-key studio lighting with a single soft key light from the upper left and a subtle cool rim light, dark neutral charcoal background, natural skin texture with visible pores and fine detail, calm confident expression, muted natural colour grade, cinematic, high detail, vertical 3:4 composition, photography — not illustration.

**Negative prompt:** `text, watermark, logo, extra fingers, deformed hands, fused fingers, missing fingers, plastic skin, oversmoothed, beauty filter, HDR glow, oversaturated, neon, cartoon, 3d render, illustration, cgi`

### A2 — Secondary (section art, optional)

> Photorealistic close-up of a hand holding a modern smartphone at chest height as if taking a selfie, fingertips wrapped around the phone edge and clearly visible, natural window light from the side, soft shadows, neutral indoor background heavily blurred, 50mm lens, f/2.8, documentary photography style, muted colour grade, high detail, vertical 3:4.

**Negative prompt:** same as A1.

### A3 — Abstract fallback (safest, zero likeness risk)

> Extreme macro photograph of a human fingertip pressed lightly against clear glass, friction ridge pattern visible, dramatic side lighting, very shallow depth of field, dark background, scientific macro photography, high detail, no text, vertical 3:4.

**Note on likeness:** if a generated person looks like a real identifiable individual, discard it. Prefer A3 if you want to avoid the question entirely.

---

## B. Analysed — real capture shot list (you shoot these)

Your own hand, or a consenting adult who has agreed in writing. Shoot **RAW or max-quality JPEG**, no beauty filters, no AI "enhance", no portrait mode. Record the camera model, lens / focal length, and the measured distance for every shot — the study needs them.

### B1 — Pipeline demo input (replaces the synthetic landing frames)
- One hand, fingertips toward camera, **tack sharp**, fingertip filling a large part of the frame.
- Even, diffuse light (window light or a softbox). No harsh specular hotspot on the fingertip.
- Plain, contrasting background (matte dark or matte light — not skin-coloured).
- Distance ~20–30 cm with the main camera, or ~60 cm with a 3× telephoto.
- **Deliver:** 3–5 takes so we can pick the sharpest.

### B2 — Distance ladder (this one is the research figure)
Same hand, same pose, same camera, same lighting. **Tripod, fixed focus plane on the fingertip.** One frame at each measured distance:

`20 cm · 35 cm · 50 cm · 75 cm · 1.0 m · 1.5 m · 2.0 m · 3.0 m`

Then repeat the whole ladder with the **telephoto lens** if the phone has one. Label files `main_20cm.jpg`, `tele_150cm.jpg`, etc. This is what empirically tests the "1.5 metres" claim with a real camera instead of a model.

### B3 — Condition variations (robustness evidence)
Same framing as B1, varying one thing at a time:
- lighting: bright daylight / dim indoor / direct harsh flash / strong backlight
- motion: static on tripod vs deliberate handheld shake
- skin: dry / slightly moist / after hand cream
- **skin tone diversity** — at least 3 different consenting subjects if possible. The report needs a per-subgroup note, and "we only tested one skin tone" is a fair criticism.

### B4 — Short video clips (for the fusion result)
5–10 s handheld clips, fingertip held roughly still, natural hand tremor, no digital stabilisation if it can be turned off. One clip at ~50 cm and one at ~1 m. These feed `python -m gattaca consolidate`.

### Consent
For anyone who is not you, get a short written note: *"I consent to photographs of my hands being used for the GATTACA biometric-privacy research project, including publication of derived figures."* Keep it with the files. Do not photograph strangers or use found images.

---

## Where each file goes

```
data/own/          B1–B4 originals (git-ignored; never commit)
service/web/assets/  decorative A1/A3 + regenerated stage frames
```

Regenerate the landing stage frames from a real B1 photo with:

```bash
python scripts/make_landing_assets.py --photo data/own/B1_best.jpg
```
