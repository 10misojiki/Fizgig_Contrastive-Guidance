# MiniMax H3 Fast+ Guidance — image-only quality experiment

Fast+ keeps the thing Fizgig is unusually good at — training MiniMax H3 from ordinary still
images — and adds protection for H3's guidance-distilled output field.

It deliberately does **not** turn Fizgig into a video trainer recipe. Captioned still-image steps
get the new objective; video, voice, audio and caption-dropout steps keep their established loss.

## Windows 11 / RTX 4090 quick start

1. Install and launch Fizgig normally with `install_fizgig.bat` and `run_fizgig.bat`.
2. Select **MiniMax H3**.
3. Run **Cache Training Data** again. Current text caching always writes the empty-prompt
   embedding Fast+ needs; old caches may predate it.
4. Load **🧪 MiniMax H3 Fast+ Guidance (LoRA 8)**.
5. Leave **Optimised Likeness Learning** on for a person/character. For style, start from the
   Style preset instead and turn guidance protection on manually.
6. Keep **Reference distillation off** for this first experiment.
7. Close ComfyUI before training so Auto can use the 4090's full 24 GB VRAM.

The Fast+ preset starts here:

| Setting | Fast+ value |
|---|---|
| Dataset | still images only |
| Network | LoRA rank/alpha 8 |
| Learning rate | `1e-4` |
| Optimised Likeness | on (`20-49`) |
| Guidance form | Contrastive |
| Guidance scale | `3.5` |
| Guidance schedule | Sigma |
| Timestep sampling | Fizgig structure (Sigmoid remains an A/B option) |
| Reference distillation | off |
| Optimizer | AdamW, `weight_decay=1e-4` default |

Contrastive Guidance adds one full no-grad H3 forward to every captioned image step. On a 4090,
plan for roughly **1.5–2× the time per image step**. It still avoids video-frame token cost and
retains image-only training's main speed advantage.

## What the loss does

Let `v` be the ordinary flow target, `u` H3's prediction for the same noised latent with an empty
prompt, and `g` the effective guidance scale. Fast+ trains the prompt prediction toward:

```text
guided_target = u + g * (v - u)
```

With the Sigma schedule:

```text
g = 1 + (configured_scale - 1) * sigma
```

The extrapolation therefore fades toward ordinary flow loss near the clean end. The empty branch
runs under `no_grad`, but with the live training LoRA active, matching the reference trainer's
default null source. Both forwards receive the exact same noised video latent and silent-audio
noise rows; otherwise their difference would contain random soundtrack noise instead of prompt
guidance.

Caption-dropout steps skip the guidance correction because their prompt is already empty. This
matches the current Akane H3 behavior and prevents an empty-vs-empty contrast from being amplified.

## The useful A/B, not a kitchen-sink run

Use the same dataset, seed, captions, preview prompts and epoch checkpoints:

1. **A — matched control:** load Fast+, turn guidance protection **off**, keep Fizgig structure.
2. **B — Guidance only:** reload Fast+ unchanged (guidance on, Fizgig structure).
3. **C — Guidance + Sigmoid:** start from B and select **Sigmoid (A/B)**.

Compare equal **gradient steps**, not only equal wall time. Check a checkpoint around 400 steps
and continue only if likeness or flexibility is still improving. Render all checkpoints with the
same workflow, seed and LoRA strength.

Loading Fast+ before all three arms keeps rank, LR (`1e-4`), blocks, seed and density identical;
the most informative comparison, A → B, therefore isolates distillation protection. B → C then
asks whether Sigmoid helps this dataset instead of assuming it always does. The existing Fast
preset remains a useful production baseline, but it uses `2e-4`, so it is not the clean control
for this particular loss comparison.

## CLI equivalent

The GUI launches the same native trainer flags:

```powershell
.\venv\Scripts\python.exe src\fizgig\scripts\minimax_train.py `
  --dit "D:\models\minimax_h3_fl2va_pruned_int8_convrot.safetensors" `
  --dataset_config ".\dataset\Fizgig_train.toml" `
  --output_dir ".\output_loras" `
  --output_name "my_h3_fastplus" `
  --network_dim 8 --network_alpha 8 `
  --learning_rate 1e-4 --max_train_epochs 50 `
  --optimizer_type adamw --no_train_adaln `
  --photo_blocks "20-49" `
  --guidance_distillation_scale 3.5 `
  --guidance_loss_form contrastive `
  --guidance_loss_schedule sigma `
  --shift 0.666667
```

The value `--shift 0.666667` is the preset's 60% clean-end Fizgig structure. For the C run,
replace it with `--shift sigmoid`. For the matched A run, keep `--shift 0.666667` and remove
the three `--guidance_*` flags. Omitting `--shift` means H3's very different shift-12
model-default schedule.

## Intentionally not stacked yet

- **Ostris training adapter:** promising, but its interaction with Fizgig's quantized ConvRot
  base and the new live-null contrastive branch needs a separate measured A/B. Fast+ does not
  download or silently merge it.
- **Reference distillation:** a different teacher objective. The GUI and trainer keep it mutually
  exclusive with guidance protection so the first result is interpretable.
- **Automagic V3 and spatial-density jitter:** optimizer/geometry experiments, not fixes for the
  guidance-distillation objective. They can be evaluated after A/B/C identifies whether Fast+
  actually raises quality on the target dataset.

If training stops with an empty-prompt-cache message, run text caching once more. No recaptioning
or latent recache is required unless the dataset itself changed.

The objective follows the current [AI Toolkit guidance-loss path](https://github.com/ostris/ai-toolkit/blob/main/extensions_built_in/sd_trainer/SDTrainer.py), the [Akane H3 formulation](https://github.com/AkaneTendo25/musubi-tuner/blob/minimax-h3/src/musubi_tuner/minimax_h3/training.py), and the [diffusion-pipe H3 recommendation](https://github.com/tdrussell/diffusion-pipe/blob/main/docs/minimax_h3_notes.md), adapted to Fizgig's `x0 - noise` output sign and packed silent-audio rows.

---

## 日本語メモ

これは「H3に元から焼き込まれているガイダンス性能を、普通の画像学習で壊しにくくする」
ための機能です。Fizgigの画像1枚だけで学習できる仕組みと、人物用ブロック`20-49`はそのまま
残しています。動画や音声データを追加する必要はありません。

Windows 11 / RTX 4090では次の順で使います。

1. `install_fizgig.bat`でインストールし、`run_fizgig.bat`で起動します。
2. Base Modelで **MiniMax H3** を選びます。
3. 学習データのキャッシュをもう一度実行します。Fast+が使う空プロンプト埋め込みもここで作られます。
4. **🧪 MiniMax H3 Fast+ Guidance (LoRA 8)** を読み込みます。
5. 人物・キャラクターなら **Optimised Likeness Learning** をONのままにします。
6. 最初の検証では **Reference distillationをOFF** にします。
7. ComfyUIを閉じてから学習を開始します。

Fast+は、captionのある静止画stepごとに勾配なしforwardを1回追加するため、従来Fastより遅く
なります。ただし多数の動画frameをbackpropするわけではなく、画像学習の速度上の利点は残ります。

比較は次の3本で十分です。

1. Fast+を読み込み、GuidanceだけOFF（比較用）
2. Fast+そのまま（Guidance ON / Fizgig structure）
3. 2からTimestep samplingだけSigmoidへ変更

同じ画像、caption、seed、step数で比べます。4090でも1 stepは約1.5～2倍遅くなる想定ですが、
動画学習のように多数フレームをbackpropしない利点は残ります。

空プロンプトのcacheがないというエラーが出た場合は、captionを作り直す必要はありません。
テキストキャッシュだけを再実行してください。Sigmoidは固定の正解ではなく、3本目の比較用です。
