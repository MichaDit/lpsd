"""Private source candidates: split each fused eight-segment group in two."""
from pathlib import Path
import json
import hashlib
import shutil

ROOT = Path('/workspace/scratch/5346007753d8/lpsd')
HERE = Path(__file__).resolve().parent / 'four-fma'


def kernel(width):
    lanes = width // 64
    mm = f'_mm{width}'
    vector = f'__m{width}d'
    target = 'avx,fma' if width == 256 else 'avx512f'
    source = f'''static __attribute__((target("{target}"))) void dot_psd_four_fused_private(
    const double *x0, const double *x1, const double *x2, const double *x3,
    const double *qr, const double *qi, long int length, double *rr, double *ri)
{{
'''
    for b in range(4):
        source += f'    const {vector} a{b} = {mm}_set1_pd(x{b}[0]);\n'
        source += f'    {vector} r{b} = {mm}_setzero_pd(), i{b} = {mm}_setzero_pd();\n'
    source += f'''    long int j = 0;
    for (; j <= length - {lanes}; j += {lanes}) {{
        {vector} real = {mm}_loadu_pd(qr + j);
        {vector} imag = {mm}_loadu_pd(qi + j);
'''
    if width == 512:
        # GCC otherwise duplicates both coefficient loads into all four
        # pairs of memory-source FMAs. No instructions are emitted here.
        source += '        __asm__("" : "+v" (real), "+v" (imag));\n'
    for b in range(4):
        source += f'''        {{
            const {vector} value = {mm}_sub_pd({mm}_loadu_pd(x{b} + j), a{b});
            r{b} = {mm}_fmadd_pd(real, value, r{b});
            i{b} = {mm}_fmadd_pd(imag, value, i{b});
        }}
'''
    source += '    }\n'
    for b in range(4):
        for part, out in (('r', 'rr'), ('i', 'ri')):
            if width == 512:
                source += f'    {out}[{b}] = _mm512_reduce_add_pd({part}{b});\n'
            else:
                source += f'''    {{
        const __m128d half = _mm_add_pd(_mm256_castpd256_pd128({part}{b}),
                                       _mm256_extractf128_pd({part}{b}, 1));
        {out}[{b}] = _mm_cvtsd_f64(_mm_add_sd(half, _mm_unpackhi_pd(half, half)));
    }}
'''
    source += '    for (; j < length; ++j) {\n'
    for b in range(4):
        source += f'''        const double value{b} = x{b}[j] - x{b}[0];
        rr[{b}] += qr[j] * value{b}; ri[{b}] += qi[j] * value{b};
'''
    source += '    }\n}\n\n'
    return source


source = (ROOT / 'lpsd_fast/_native/fast_dft.c').read_text()
base_hash = hashlib.sha256(source.encode()).hexdigest()
manifest = {'base_native_source_sha256': base_hash, 'variants': {}}
for label, width, cutoff in (
        ('control8', None, None), ('strict', None, None),
        ('split256_large', 256, 2048), ('split256_all', 256, 128),
        ('split512_large', 512, 2048), ('selective8', None, None)):
    directory = HERE / label
    shutil.copytree(ROOT / 'lpsd_fast', directory / 'lpsd_fast',
                    ignore=shutil.ignore_patterns('__pycache__'), dirs_exist_ok=True)
    shutil.copytree(ROOT / 'lpsd/c_sources', directory / 'lpsd/c_sources', dirs_exist_ok=True)
    result = source
    if width is not None:
        offset = result.index('static __attribute__((target("avx512f"))) void dot_psd_eight_fused(')
        result = result[:offset] + kernel(width) + result[offset:]
        original = '''            dot_psd_eight_fused(
                segments[0], segments[1], segments[2], segments[3],
                segments[4], segments[5], segments[6], segments[7],
                projected_r, projected_i, segLen, rr, ri);'''
        isa = ' && __builtin_cpu_supports("fma")' if width == 256 else ''
        replacement = f'''            if (segLen >= {cutoff}{isa}) {{
                dot_psd_four_fused_private(segments[0], segments[1], segments[2], segments[3],
                                            projected_r, projected_i, segLen, rr, ri);
                dot_psd_four_fused_private(segments[4], segments[5], segments[6], segments[7],
                                            projected_r, projected_i, segLen, rr + 4, ri + 4);
            }} else {{
{original}
            }}'''
        assert original in result
        result = result.replace(original, replacement, 1)
    elif label == 'selective8':
        old = 'const bool fused_psd = batched && mode >= 2 && !csd && segLen >= 128 &&'
        new = '''const bool fused_psd = batched && mode >= 2 && !csd && segLen >= 128 &&
        (segLen < 256 || (segLen >= 1024 && segLen < 2048) ||
         (segLen >= 8192 && segLen < 65536)) &&'''
        assert old in result
        result = result.replace(old, new, 1)
    (directory / 'lpsd_fast/_native/fast_dft.c').write_text(result)
    if label == 'strict':
        api_path = directory / 'lpsd_fast/api.py'
        api = api_path.read_text()
        assert 'use_bounded = (' in api
        api = api.replace('use_bounded = (', 'use_bounded = False and (', 1)
        api_path.write_text(api)
    manifest['variants'][label] = {
        'root': str(directory), 'vector_bits': width, 'minimum_length': cutoff,
        'native_source_sha256': hashlib.sha256(result.encode()).hexdigest(),
    }
(HERE / 'prepare.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(HERE / 'prepare.json')
