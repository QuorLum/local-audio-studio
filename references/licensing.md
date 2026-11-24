# Licensing and attribution

This is a plain-language summary to help you ask the right questions. It is **not legal advice**; the authoritative texts are linked below. Read this when a user plans to publish, sell, or build a product or service on the generated audio or on this tool.

**Contents** - [Who owns what](#who-owns-what) - [Stability AI Community License](#stability-ai-community-license-sound-effects-and-music) - [Gemma Terms](#gemma-terms-text-encoder) - [Kokoro](#kokoro-speech) - [Checklist](#checklist) - [Trademarks](#trademarks)

## Who owns what

| Part | License | Where the text is |
|---|---|---|
| This repository's code (the "wrapper") | MIT | `LICENSE` |
| Kokoro-82M weights and the `kokoro` / `misaki` packages | Apache-2.0 | model card of `hexgrad/Kokoro-82M` |
| Stable Audio 3 Small SFX and Small Music **weights** | Stability AI Community License | `licenses/STABILITY_AI_COMMUNITY_LICENSE.md` |
| `stable-audio-3` inference code (installed from GitHub) | MIT | Stability-AI/stable-audio-3 |
| T5Gemma text encoder used by Stable Audio 3 | Gemma Terms of Use | https://ai.google.dev/gemma/terms |

The repository contains no weights; the setup script downloads them from their publishers (`NOTICE` lists every component). The MIT license of the wrapper does **not** relax any of the model licenses.

## Stability AI Community License (sound effects and music)

What the agreement says (see the full text for the exact wording):

* **Free for research, non-commercial and personal (hobbyist) use.**
* **Commercial use is free while you, together with your affiliates, earn less than US $1,000,000 a year.** For commercial use you must register at https://stability.ai/community-license. Above the threshold the license ends and you need an enterprise license (https://stability.ai/enterprise).
* **Outputs:** as between you and Stability AI, you own the audio you generate, to the extent the law permits.
* **No training foundation models:** you may not use the models, derivatives *or their outputs* to create or improve any foundational generative AI model (other than the Stability models and derivatives).
* **Acceptable use:** your use and the outputs must follow applicable law (including export-control and sanctions rules) and Stability's Acceptable Use Policy (https://stability.ai/use-policy).
* **If you distribute** the models, a derivative, or **a product or service that uses them**: include a copy of the agreement, keep the notice text *"This Stability AI Model is licensed under the Stability AI Community License, Copyright (c) Stability AI Ltd. All Rights Reserved"* in a Notice file, and prominently display **"Powered by Stability AI"** on a related website, user interface, blog post, about page or product documentation. If you modify the materials, say so in the Notice file.
* No trademark rights are granted beyond that attribution.

This repository's README and SKILL.md display "Powered by Stability AI", `NOTICE` carries the required text, and the vendored configs (derived files) are described in `NOTICE`.

## Gemma Terms (text encoder)

Stable Audio 3 uses Google's T5Gemma encoder to read prompts. Stability's model pages state that components redistributed with the model are under the **Gemma Terms of Use**, including its use restrictions. The encoder file is downloaded, not redistributed, by this project; by using it you accept those terms: https://ai.google.dev/gemma/terms and the linked prohibited-use policy.

## Kokoro (speech)

Apache-2.0: you may use the voices commercially, including the generated speech, without attribution of the output. If you *redistribute* the Kokoro weights or code, keep the Apache-2.0 license and notices. If you ship individual voices inside a product, read the model card and `VOICES.md` of `hexgrad/Kokoro-82M` for any dataset attributions that apply to those voices.

The English text front-end calls the espeak-ng library (GPL-3.0) as a separate shared library installed from PyPI; it is not part of this repository.

## Checklist

For **publishing audio** (videos, podcasts, games):

1. Speech from Kokoro: no restriction beyond the ordinary law (do not impersonate real people; Kokoro has no voice cloning).
2. Effects and music from Stable Audio 3: personal or non-commercial use is free. For a commercial project check the revenue test above and register once at Stability's community-license page. Keep the audio out of any dataset used to train a foundation model.
3. Do not use the audio for anything the Acceptable Use Policy forbids.

For **shipping a tool, app or service** that bundles or calls this project:

1. Provide the agreement text and the notice (`licenses/`, `NOTICE`).
2. Show "Powered by Stability AI" in your UI or documentation.
3. State your own modifications in the Notice file.
4. Decide whether your revenue puts you under the US $1M threshold, and register if the use is commercial.
5. Do not redistribute the model weights unless you are ready to meet every condition above and the Gemma terms.

## Trademarks

ElevenLabs is a trademark of its owner. This project is an independent open-source tool and is not affiliated with, endorsed by or a copy of that service; it is described as an *alternative* only. Stable Audio, Stability AI, Gemma and Kokoro are names of their respective owners.
