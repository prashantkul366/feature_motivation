"""
Frozen configuration for the KAN / Mamba(VSS) / Transformer complementarity
motivation study. Single source of truth -- nothing below is duplicated in
train.py or the model files.

CLAIM THIS EXPERIMENT SUPPORTS
  Under matched capacity and identical training, KAN, VSS and Transformer
  blocks converge to representations measurably less similar to each other
  than the same block is to itself across random seeds, at comparable
  CIFAR-100 accuracy.

CLAIM IT DOES NOT SUPPORT
  That the differences are useful, that fusing them helps, or that any of
  this transfers to segmentation. Those are the ablation section's job.
"""

# ---------------------------------------------------------------- seeds ----
SEEDS = (42, 123, 2024)          # per-run: init + data order
SPLIT_SEED = 0                   # data split + analysis subset; NEVER changes
TOKEN_PAIR_SEED = 0              # frozen (image, token) pairs for token-level analysis

# ----------------------------------------------------------- shared stem ---
IMG_SIZE = 32
PATCH_SIZE = 4                   # -> 8x8 grid = 64 tokens
N_TOKENS = 64
GRID = 8
EMBED_DIM = 192
DEPTH = 6                       # blocks per model
NUM_CLASSES = 100
IN_CHANS = 3

# ------------------------------------------------------------ block cfgs ---
# Parameter counts per block, computed against the actual source files.
# Target: all three within +/-15% of the mean. Verified by verify_parity().
VIT = dict(
    num_heads=3,                 # head_dim 64
    mlp_dim=4 * EMBED_DIM,       # 768
    dropout_rate=0.0,            # MUST stay 0: KAN/VSS have no dropout, so any
    attention_dropout_rate=0.0,  # nonzero value is a straight confound
    layer_norm_eps=1e-6,
)

VSS = dict(
    ssm_ratio=1.0,               # REQUIRED for parity. VMamba's default 2.0
                                 # makes in_proj Linear(192,768) and blows the
                                 # block to 636K (+43% vs ViT).
    ssm_d_state=16,
    ssm_conv=3,                  # depthwise conv: VSS is not purely SSM
    ssm_dt_rank="auto",          # -> ceil(192/16) = 12
    mlp_ratio=4.0,
    forward_type="v05",          # no_einsum, force_fp32=False, backend=None
                                 # -> falls back to selective_scan_torch when
                                 # no CUDA extension is built (i.e. on Colab)
    channel_first=False,         # expects (B, H, W, C)
    drop_path=0.0,
    ssm_drop_rate=0.0,
    mlp_drop_rate=0.0,
    post_norm=False,             # pre-norm, matching ViT
    force_torch_scan=True,       # bypass Triton for cross_scan/cross_merge: at
                                 # 8x8 the kernel uses 32x32 blocks, 64 valid
                                 # lanes of 1024. NOTE this controls the CROSS
                                 # SCAN only, not the selective scan below.

    # Selective-scan implementation. MUST be set explicitly, because csms6s
    # auto-selects: merely pip-installing mamba_ssm into the session flips
    # this from the torch reference to a CUDA kernel with no change to this
    # repo, and nothing would look wrong. Run check_scan_backends.py first
    # and record whichever you pin here in the paper.
    #   "auto"     - csms6s decides (CUDA if importable, else torch)
    #   "torch"    - selective_scan_torch, the reference. ~172 img/s.
    #   "parallel" - prefix scan, verified equivalent to the reference at
    #                3e-7 relative on both forward and gradients
    #   "cuda"     - force the mamba_ssm kernel
    selective_scan_impl="auto",
)

KAN = dict(
    grid_size=5,                 # LOAD-BEARING for parity: grid_size=7 pushes
                                 # the channel-mix layer to 442K and breaks it
    spline_order=3,
    grid_range=(-2.0, 2.0),      # NOT the efficient-kan default of (-1,1).
                                 # LayerNorm gives unit variance, so ~32% of
                                 # activations would fall outside +/-1 where the
                                 # spline branch is inactive and KANLinear
                                 # degrades toward SiLU+Linear, i.e. an MLP.
    base_activation="silu",
    token_mix=True,              # Mixer-style: KANLinear(64->64) across tokens
                                 # THEN KANLinear(192->192) across channels.
                                 # Without token mixing KAN cannot move
                                 # information spatially while ViT and VSS can,
                                 # which would rig the comparison.
    update_grid=False,           # adaptive grids break seed reproducibility
)

# --------------------------------------------------------------- training --
TRAIN = dict(
    epochs=100,
    batch_size=128,              # if OOM (L4/T4 instead of A100): drop to 64
    lr=1e-3,                     # AND lr to 7e-4, FOR ALL THREE MODELS
    min_lr=1e-5,
    warmup_epochs=5,
    weight_decay=0.05,
    label_smoothing=0.1,
    grad_clip=1.0,
    optimizer="adamw",
    scheduler="cosine",
    amp=False,                   # fp32 throughout. selective_scan_torch hard-
                                 # casts to fp32 internally and KAN's b_splines
                                 # needs fp32, so AMP would speed up ONLY ViT
                                 # -- a precision asymmetry in an experiment
                                 # whose validity rests on the three models
                                 # being identical apart from the block.
    num_workers=2,
)

# augmentation: deliberately minimal. mixup/cutmix would raise accuracy but
# reshape representation geometry, forcing us to argue the similarity numbers
# are not an augmentation artefact.
AUGMENT = dict(random_crop_padding=4, random_hflip=True)

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)

# --------------------------------------------------------------- analysis --
ANALYSIS = dict(
    n_examples=2000,
    extract_after_blocks=(1, 2, 3),   # headline is block 3
    pool="mean",                      # over the 64 tokens
    also_token_level=True,            # ALSO compare unpooled [N*64, 192].
                                      # VSS's mechanism is directional scanning
                                      # over 2D space; mean-pooling collapses
                                      # spatial arrangement and may wash out
                                      # exactly what we are trying to measure.
    svcca_components=50,
    svcca_robustness=(30, 50, "99pct"),
    cka_headline="debiased",          # biased CKA has a floor of 0.089 (linear)
                                      # / 0.141 (RBF) at N=2000, D=192.
                                      # Measured in test_similarity.py CASE 5.
)

# Empirical noise floors at our operating point (N=2000, D=192), measured on
# independent gaussians. Real numbers must be read against these, not against 0.
NOISE_FLOOR = dict(
    svcca_k50=0.135,
    cka_linear_biased=0.089,
    cka_rbf_biased=0.141,
    cka_debiased=0.001,
)

# ------------------------------------------------------------------ gates --
# Pre-declared before any run. Committing to these now is what stops the
# analysis drifting into confirmation-seeking later.
GATES = dict(
    param_parity_tolerance=0.15,      # max deviation from mean param count
    accuracy_gap_clean=3.0,           # pp: write "comparable performance"
    accuracy_gap_acceptable=6.0,      # pp: report the gap, soften wording
    # above 6pp: ONE pre-declared LR sweep {5e-4, 1e-3, 2e-3} applied to ALL
    # THREE models, best-val-picked, disclosed. If still >6pp, drop the
    # parity clause rather than tuning until it fits.
    lr_sweep_if_needed=(5e-4, 1e-3, 2e-3),
    # KILL CRITERION: any cross-arch SVCCA within 1 std of the corresponding
    # within-arch (seed control) value means the architectural difference is
    # not distinguishable from initialisation noise. The motivation claim must
    # then be dropped or heavily qualified.
)

# Expected CIFAR-100 top-1 for a 3-block model at this scale is 40-55%.
# NOT 87%. The source paper's 87.8% is not CIFAR-100 top-1 at this depth.
EXPECTED_TOP1_RANGE = (0.40, 0.55)