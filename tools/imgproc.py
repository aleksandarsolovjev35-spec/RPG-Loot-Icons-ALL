"""Image-processing primitives for icon repacking (numpy + Pillow only).

The 512 px set is a 3.46x upscale of a ~148 px crop that arrived as JPEG, so
every artefact of the chain gets magnified:

* JPEG mosquito noise / 8 px blockiness in gradients and near contours;
* ringing of the resampler: a bright fringe just inside a silhouette drawn on
  the black background (Lanczos overshoot) and a dark fringe outside it;
* the pixel staircase of the 148 px artwork, stretched to ~3.5 px steps;
* WebP quantisation noise on top of all of it.

The helpers here attack those four separately and are deliberately free of
scipy/OpenCV so the toolchain stays `pip install pillow numpy`:

    resample_ar()        separable resample with an anti-ringing clamp - keeps
                         the edge position of Lanczos but removes the overshoot;
    guided()             edge-preserving low-pass (He et al.) used both to
                         denoise the JPEG and to clean up after each step;
    steered_blur()       blurs *along* the local contour and is range-weighted
                         *across* it: rounds the staircase, leaves the contour
                         profile alone;
    masked_unsharp()     unsharp with a soft-threshold (noise stays put) and a
                         hard clamp (no overshoot) - the "improve the contours"
                         part without halos;
    black_floor()        the sheets sit on pure black: flatten the last bit of
                         JPEG mottle to 0 so the background is dead flat.
"""
import numpy as np
from PIL import Image

RGB_W = np.array([0.299, 0.587, 0.114], np.float32)

# --------------------------------------------------------------------------
# basics
# --------------------------------------------------------------------------


def luma(x):
    return x @ RGB_W


def soft(x, lo, hi):
    """smoothstep ramp: 0 below lo, 1 above hi."""
    t = np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def to_u8(x):
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


def to_f(img):
    """uint8 image / array -> float32 in [0, 1]."""
    return np.asarray(img, np.float32) / 255.0


def box(img, r):
    """Mean filter of radius r (reflect padding, integral image)."""
    if r < 1:
        return img.copy()
    a = np.pad(img, ((r, r), (r, r)) + ((0, 0),) * (img.ndim - 2), mode='reflect')
    c = np.cumsum(np.cumsum(a, 0, dtype=np.float32), 1, dtype=np.float32)
    c = np.pad(c, ((1, 0), (1, 0)) + ((0, 0),) * (img.ndim - 2), mode='constant')
    H, W = img.shape[:2]
    k = 2 * r + 1
    s = c[k:k + H, k:k + W] - c[0:H, k:k + W] - c[k:k + H, 0:W] + c[0:H, 0:W]
    return s / float(k * k)


def _gauss_kernel(sigma):
    r = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def convolve_axis(a, k, axis):
    """Separable 1-D convolution along `axis` (reflect padding, float32)."""
    r = (len(k) - 1) // 2
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r, r)
    p = np.pad(a, pad, mode='reflect')
    n = a.shape[axis]
    idx = np.arange(n)
    out = None
    for i, ki in enumerate(k):
        if ki == 0.0:
            continue
        sl = np.take(p, idx + i, axis=axis)
        out = sl * ki if out is None else out + sl * ki
    return out


def blur(img, sigma):
    """Separable gaussian blur (reflect padding)."""
    if sigma <= 0:
        return img.copy()
    k = _gauss_kernel(sigma)
    return convolve_axis(convolve_axis(img, k, 0), k, 1)


# --------------------------------------------------------------------------
# edge preserving low pass
# --------------------------------------------------------------------------


def guided(src, guide, r=3, eps=1e-3):
    """Guided filter (He, Sun, Tang 2010), float in/out.

    `guide` is a 2-D luma field; `src` may be HxW or HxWx3.  Trust region is
    chosen by the guide, so edges of the artwork survive while grain between
    them is averaged away.
    """
    g = guide[..., None]
    m_g, m_s = box(g, r), box(src, r)
    var = box(g * g, r) - m_g * m_g
    cov = box(g * src, r) - m_g * m_s
    a = cov / (var + eps)
    b = m_s - a * m_g
    return box(a, r) * g + box(b, r)


def denoise(x, eps=0.02, r=2, passes=1):
    """JPEG-artefact cleanup: guided smoothing guided by its own luma."""
    for _ in range(passes):
        x = guided(x, luma(x), r=r, eps=eps * eps)
    return np.clip(x, 0.0, 1.0)


def black_floor(x, knee=0.012):
    """Sheets sit on black - pull the last smear of JPEG mottle down to 0."""
    if not knee:
        return x
    return x * soft(luma(x), 0.0, knee)[..., None]


# --------------------------------------------------------------------------
# resampling with an anti-ringing clamp
# --------------------------------------------------------------------------
KERNELS = {
    # name: (callable u -> weight, radius)
    'lanczos3': (lambda u: np.sinc(u) * np.sinc(u / 3.0), 3.0),
    'lanczos2': (lambda u: np.sinc(u) * np.sinc(u / 2.0), 2.0),
    'catmullrom': (None, 2.0),
    'mitchell': (None, 2.0),
    'bilinear': (None, 1.0),
}


def _kernel_weights(u, name):
    a = KERNELS[name][1]
    inside = np.abs(u) < a
    if name.startswith('lanczos'):
        w = np.sinc(u) * np.sinc(u / a)
    elif name == 'catmullrom':
        au = np.abs(u)
        w = np.where(au < 1, 1.5 * au ** 3 - 2.5 * au ** 2 + 1,
                     np.where(au < 2, -0.5 * au ** 3 + 2.5 * au ** 2 - 4 * au + 2, 0.0))
    elif name == 'mitchell':
        au = np.abs(u)
        w = np.where(au < 1, (7 * au ** 3 - 12 * au ** 2 + 16.0 / 3) / 9.0,
                     np.where(au < 2, (-7.0 / 3 * au ** 3 + 12 * au ** 2 - 20 * au + 32.0 / 3) / 9.0, 0.0))
    elif name == 'bilinear':
        w = np.maximum(1.0 - np.abs(u), 0.0)
    else:
        raise ValueError(name)
    return w * inside


def _axis_weights(n_in, n_out, name, stretch):
    """weights[j, t], taps[j, t] for one axis (weights already normalised)."""
    scale = float(n_in) / n_out
    # stretch=True keeps the true kernel width in source pixels (a*scale, i.e.
    # a proper sinc-like reconstruction); stretch=False is the Pillow
    # convention - kernel never narrower than a source pixel, which reads a bit
    # softer because more neighbours vote.  Downscaling always stretches.
    fs = scale if (stretch or scale > 1.0) else 1.0
    support = KERNELS[name][1] * fs
    ntap = int(np.floor(2.0 * support)) + 2
    j = np.arange(n_out, dtype=np.float32)
    center = (j + 0.5) * scale - 0.5
    t = np.arange(ntap, dtype=np.float32)
    taps = (np.floor(center - support) + 1.0)[:, None] + t[None, :]
    rel = taps - center[:, None]
    w = _kernel_weights(rel / fs, name)
    w = np.where(np.abs(rel) <= support + 1e-6, w, 0.0).astype(np.float32)
    s = w.sum(1, keepdims=True)
    s[s == 0] = 1.0
    w /= s
    return w, np.clip(taps, 0.0, n_in - 1.0).astype(np.int32)


def _apply_axis(src, w, taps, axis, clamp, chunk=64):
    """Resample one axis of an HxW(,C) float array; `axis` is the source axis."""
    s = np.moveaxis(src, axis, 0)                       # (n_in, rest...)
    n_out = w.shape[0]
    shape = (n_out,) + s.shape[1:]
    out = np.empty(shape, np.float32)
    rest = int(np.prod(s.shape[1:], dtype=np.int64))
    s2 = s.reshape(s.shape[0], rest)
    o2 = out.reshape(n_out, rest)
    step = max(1, int(chunk * 4096 // max(1, taps.shape[1] * n_out)))
    for a in range(0, rest, step):
        b = min(rest, a + step)
        stack = s2[taps, a:b]                           # (n_out, ntap, blk)
        acc = np.einsum('jt,jtb->jb', w, stack, optimize=True)   # (n_out, blk)
        if clamp:
            acc = np.clip(acc, stack.min(1), stack.max(1))
        o2[:, a:b] = acc
    return np.moveaxis(out, 0, axis)


def resample_ar(src, size, kernel='lanczos3', clamp=True, stretch=False):
    """Resample a float HxW(,C) image to `size` (int or (w, h)).

    With `clamp` the result can never overshoot the range of the samples that
    produced it: that is what removes the halo Lanczos leaves around a bright
    silhouette drawn on black - the main reason the current 512 set looks
    "glowy" at 1:1.
    """
    if np.isscalar(size):
        size = (size, size)
    out = np.ascontiguousarray(src, np.float32)
    # pass 1: x (shape[1] -> size[0]); pass 2: y (shape[0] -> size[1])
    for axis, (n_in, n_out) in ((1, (out.shape[1], size[0])), (0, (out.shape[0], size[1]))):
        if n_in == n_out:
            continue
        w, taps = _axis_weights(n_in, n_out, kernel, stretch)
        out = _apply_axis(out, w, taps, axis=axis, clamp=clamp)
    return out


# --------------------------------------------------------------------------
# contour work on the upscaled image
# --------------------------------------------------------------------------


def orientation(L, rho=2.0, extra=0.8):
    """Structure tensor of the luma field.

    Returns (l1, coherence^2, nx, ny) where (nx, ny) is the eigenvector ACROSS
    the contour (the l1 one) and `coherence` measures how line-like the
    neighbourhood is - 1 on a clean contour, ~0 on noise and corners.
    """
    gy, gx = np.gradient(L)
    Jxx, Jyy, Jxy = blur(gx * gx, rho), blur(gy * gy, rho), blur(gx * gy, rho)
    if extra:
        Jxx, Jyy, Jxy = blur(Jxx, extra), blur(Jyy, extra), blur(Jxy, extra)
    tr = Jxx + Jyy
    disc = np.sqrt(np.maximum(tr * tr * 0.25 - (Jxx * Jyy - Jxy * Jxy), 0.0))
    l1, l2 = tr * 0.5 + disc, tr * 0.5 - disc
    vx, vy = Jxy, l1 - Jxx
    n = np.hypot(vx, vy)
    deg = n < 1e-9
    vx = np.where(deg, 1.0, vx / np.where(deg, 1.0, n))
    vy = np.where(deg, 0.0, vy / np.where(deg, 1.0, n))
    coh = (l1 - l2) / (l1 + l2 + 1e-9)
    return l1, coh * coh, vx, vy


def sample_bilinear(img, xs, ys):
    H, W = img.shape[:2]
    x0 = np.floor(xs)
    y0 = np.floor(ys)
    fx = (xs - x0)
    fy = (ys - y0)
    if img.ndim == 3:
        fx = fx[..., None]
        fy = fy[..., None]
    x0 = np.clip(x0, 0, W - 2).astype(np.int32)
    y0 = np.clip(y0, 0, H - 2).astype(np.int32)
    c00, c10 = img[y0, x0], img[y0, x0 + 1]
    c01, c11 = img[y0 + 1, x0], img[y0 + 1, x0 + 1]
    top = c00 + (c10 - c00) * fx
    bot = c01 + (c11 - c01) * fx
    return top + (bot - top) * fy


def steered_blur(x, sigma=3.0, taps=5, rho=2.0, range_knee=0.030, coh_min=0.05):
    """Blur along the contour, weight by colour similarity across it.

    This is the "de-staircase" step: the ~3.5 px steps the upscale makes out of
    the 148 px artwork get averaged along the contour direction, so the edge
    becomes a smooth curve, while the across-edge profile is protected by the
    range weight (a neighbouring pixel only contributes when it has about the
    same colour) and by the coherence gate (flat/noisy places are left alone).
    """
    L = luma(x)
    _l1, coh2, nx, ny = orientation(L, rho=rho)
    tx, ty = -ny, nx
    H, W = L.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    num = np.zeros_like(x)
    den = np.zeros((H, W, 1), np.float32)
    for t in range(-taps, taps + 1):
        if t == 0:
            continue
        w = float(np.exp(-0.5 * (t / sigma) ** 2))
        if w < 0.01:
            continue
        s = sample_bilinear(x, xx + t * tx, yy + t * ty)
        rw = np.exp(-((luma(s) - L) / range_knee) ** 2)
        ww = (w * rw * np.clip(coh2 / max(coh_min, 1e-6), 0.0, 1.0))[..., None]
        num += s * ww
        den += ww
    return (x + num) / (1.0 + den)


def masked_unsharp(x, sigma=1.6, amount=0.30, thr=0.010, knee=0.04, clamp=0.05):
    """Unsharp with a soft-threshold and a clamp - crispness without halos.

    Only detail above `thr` is amplified (so grain in the flats is left alone),
    the ramp to full gain ends at `knee`, and the added amount is capped at
    `clamp` (0.05 = 13 levels of 255) so a contour cannot grow a bright rim.
    """
    if not amount:
        return x
    d = x - blur(x, sigma)
    w = soft(np.abs(d), thr, knee)
    return np.clip(x + np.clip(d * amount * w, -clamp, clamp), 0.0, 1.0)


def keep_black(out, src, knee=0.04):
    """Don't lift pixels that were black in `src` — kills silhouette halos."""
    if not knee:
        return out
    w = (1.0 - soft(luma(src), 0.0, knee))[..., None]
    return out * (1.0 - w) + src * w


def laplacian_detail(x, amount=0.70, fine=0.06, coarse=0.22,
                     s1=1.0, s2=2.6, s3=6.5):
    """Multi-scale detail without razor contours.

    The mid band (blur s1..s2) is remapped: small coefficients (texture, facets,
    folds) get `amount` extra gain, large ones (the silhouette) stay near 1.
    Fine/coarse bands get a light interior-only lift.  Empty black is pinned
    to the pre-filter image so a glow cannot grow into the background.
    """
    if not amount:
        return x
    g1, g2, g3 = blur(x, s1), blur(x, s2), blur(x, s3)
    fine_b, mid, coarse_b, base = x - g1, g1 - g2, g2 - g3, g3
    mag = np.mean(np.abs(mid), axis=-1, keepdims=True)
    gain = 1.0 + amount * (1.0 - soft(mag, 0.025, 0.10))
    inter = soft(luma(x), 0.04, 0.12)[..., None]
    y = (base
         + coarse_b * (1.0 + coarse * inter)
         + mid * (1.0 + (gain - 1.0) * inter)
         + fine_b * (1.0 + fine * inter))
    return keep_black(np.clip(y, 0.0, 1.0), x)


# --------------------------------------------------------------------------
# whole pipeline
# --------------------------------------------------------------------------
DEFAULTS = dict(size=512, denoise=0.014, denoise_r=1, black=0.010,
                kernel='lanczos3', clamp=True, stretch=False,
                steer=3.0, steer_taps=5, steer_rho=2.0, steer_range=0.030,
                sharpen=0.30, sharpen_sigma=1.6, sharpen_thr=0.010, sharpen_knee=0.04,
                sharpen_clamp=0.05, laplacian=0.0, post_denoise=0.0, native_u8=True)


def enhance(crop_u8, cfg=None):
    """Native RGB crop (uint8 HxWx3) -> enhanced square uint8 image."""
    c = dict(DEFAULTS)
    c.update(cfg or {})
    x = to_f(crop_u8)
    if c['denoise']:
        x = denoise(x, eps=c['denoise'], r=c['denoise_r'])
    if c['black']:
        x = black_floor(x, c['black'])
    if c['native_u8']:
        x = to_f(to_u8(x))
    if c['size']:
        u = resample_ar(x, c['size'], kernel=c['kernel'], clamp=c['clamp'], stretch=c['stretch'])
    else:
        u = x
    if c['steer']:
        u = np.clip(steered_blur(u, sigma=c['steer'], taps=c['steer_taps'], rho=c['steer_rho'],
                                 range_knee=c['steer_range']), 0.0, 1.0)
    if c['sharpen']:
        u = masked_unsharp(u, sigma=c['sharpen_sigma'], amount=c['sharpen'],
                           thr=c['sharpen_thr'], knee=c['sharpen_knee'],
                           clamp=c['sharpen_clamp'])
    if c['laplacian']:
        sc = (c['size'] / 256.0) if c.get('size') else 1.0
        u = laplacian_detail(u, amount=c['laplacian'],
                             s1=1.0 * sc, s2=2.6 * sc, s3=6.5 * sc)
    if c['post_denoise']:
        u = denoise(u, eps=c['post_denoise'])
    return to_u8(u)
