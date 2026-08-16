# Chapter 17 programs

Generative adversarial networks.  Five modules; the first three are pure
NumPy/autograd and reproduce every number quoted in `doc/BookML/chapter17.tex`,
the last two are the MNIST programs in PyTorch and TensorFlow.

| file | what it does |
|------|--------------|
| `gan.py` | networks, the three generator losses, the WGAN gradient penalty, Adam with `beta_1 = 0.5`, the alternating training loop, the 2-D targets and the energy distance |
| `verify_gan.py` | the eight checks of the chapter -> `verify.txt` |
| `run_compare.py` | equilibrium, gradient starvation and mode collapse -> `compare.txt` and the `.npy` arrays the figures use |
| `gan_torch.py` | MNIST GAN in PyTorch: `--arch fc/dcgan`, `--loss nonsat/sat/wgangp`, `--smooth 0.9` |
| `gan_tf.py` | the same program in TensorFlow/Keras |

Run order:

```bash
python verify_gan.py       # ~40 s
python run_compare.py      # ~4 min
cd ../../BookFigures && CH17_RUN=../BookPrograms/chapter17_gan python ch17_figures.py
```

`verify_gan.py` and `run_compare.py` need `autograd`, `numpy` and `scipy`;
`ch17_figures.py` needs `matplotlib`.
