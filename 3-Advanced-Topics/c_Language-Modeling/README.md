# Language Modeling: nanoGPT in MATLAB

Build a generative pretrained transformer (GPT) from scratch in MATLAB&reg;, following Andrej Karpathy's ["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) lecture. Starting from a completely random model, the script incrementally adds complexity &mdash; bigram statistics, self-attention, multi-head attention, feedforward layers, and residual connections &mdash; until the model generates recognizable text.

## Requirements

| Product | Required |
|---|---|
| [MATLAB&reg;](https://www.mathworks.com) R2023b+ | Yes |
| [Deep Learning Toolbox&trade;](https://www.mathworks.com/products/deep-learning.html) | Yes |
| [Text Analytics Toolbox&trade;](https://www.mathworks.com/products/text-analytics.html) | Yes |
| [Parallel Computing Toolbox&trade;](https://www.mathworks.com/products/parallel-computing.html) | For GPU training |

## Getting Started

Open `BuildNanoGPT.m` and run it. The script trains a series of progressively larger models on a character-level language modeling task.

Two training datasets are included in `data/`:

| Dataset | Language | Description |
|---|---|---|
| `tinyshakespeare.txt` | English | Shakespeare's collected works |
| `quijote.txt` | Spanish | Don Quijote de la Mancha |

Pre-trained models for both datasets are available in `models/`.

## Results

View the full pre-generated HTML reports for each dataset:

* [Build NanoGPT &mdash; Tiny Shakespeare](https://mathinking.github.io/dl-exploration-lab/3-Advanced-Topics/c_Language-Modeling/results/BuildNanoGPT_tinyshakespeare.html)
* [Build NanoGPT &mdash; Don Quijote](https://mathinking.github.io/dl-exploration-lab/3-Advanced-Topics/c_Language-Modeling/results/BuildNanoGPT_quijote.html)

### Sample output &mdash; Tiny Shakespeare

After training the final model (6 heads, 6 blocks, 384-dim embeddings, context length 256) for 5000 iterations (~36 min on an NVIDIA RTX 3500 Ada laptop GPU):

```
Be a ready good mercy, not to fear: then, friend of the law,
confusion the noble Paris shall do lift thine.

GREEN:
So far up your household.

GREGORY:
Hath he for fourther blessed them to for fell dreams,
But thou wouldst be the bastard.

SAMPSON:
For us they are drawn ours: yet may you, my demanders,
The meaner to the justice of fair usurprise.

GREGORY:
You may be entreaten.

SAMPSON:
Why, my lord; you must be, sir, I wish.

GREGORY:
```

### Sample output &mdash; Don Quijote

After training the same architecture on the Spanish text for 5000 iterations (~71 min on an NVIDIA RTX 3500 Ada laptop GPU):

```
ue traía de su amo se habían menester de dar un alma. Don Quijote, que
encomiéndolas partió todo Sancho Panza a su padre, y, diciendo esto aviso
que no pudiese poder dejar entender otro reliego de la roca a oro, y aún
no le quedaremos, como tú industes, o de reverencia en aquel costal: que
es mi cargo por osmayar, mira la bendición descomuna.

-Aun esto -replicó Sancho-, que vos, a ser qué se lo ofrece, porque yo
la lo he de amar de divinar, y el cura conocimiendo, no queme en poder,
confiarlas, finalmente, habéis acabado, que es de serviros con apriesa lo
que dél se vieran a los puntos cande angustianos, quedaban sus jugas, en
esta aventura
```

---

Copyright 2024-2026 The MathWorks, Inc.
