"""Freeze a new decoder-only configuration after explicit original-VAE preparation."""
import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h3_t8.freevideo_decoder import contract as c
from h3_t8.freevideo_exp.runtime import canonical, digest, verify_files, write_json_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('quality-runtime', 'assets-proof', 'home', 'output-config'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    output = args.output_config.resolve()
    if output.exists() or args.home.exists():
        raise FileExistsError('Create a new decoder config and cache home; never replace a sampler runtime')
    runtime = c.read_json(args.quality_runtime)
    if (runtime.get('schema') != 't8-freevideo-runtime-v2'
            or runtime.get('freevideo_revision') != c.FREEVIDEO_REVISION
            or runtime.get('vdn_revision') != c.VDN_REVISION
            or runtime.get('inventory_sha256') != hashlib.sha256(canonical(runtime['files']).encode()).hexdigest()):
        raise ValueError('Borrow only the current pinned v0.2.3 source and actual dependency inventory')
    assets = c.read_json(args.assets_proof)
    if (assets.get('schema') != 't8-freevideo-decoder-assets-v1'
            or assets.get('model_revision') != c.VAE_REVISION or assets.get('repository') != 'MiniMaxAI/MiniMax-H3'
            or assets.get('originals_not_converted') is not True or len(assets.get('files', [])) != 5):
        raise ValueError('Use the verified original official VAE assets, not a renamed Comfy VAE')
    base = args.assets_proof.resolve().parent
    expected = {str(base / 'vae' / name): (size, sha, 'vae/' + name) for name, (size, sha) in c.ASSETS.items()}
    observed = {row['path']: (row['bytes'], row['sha256'], row['source']) for row in assets['files']}
    if observed != expected or len(observed) != 5:
        raise ValueError('Asset proof does not match all five pinned original files')
    # Read-only reuse of code/dependencies, never the unrelated DiT or AdaLN weight files.
    rows = [dict(path=row['path'], bytes=row['bytes'], sha256=row['sha256'], kind='borrowed_source_dependency')
            for row in runtime['files'] if row['kind'] != 'weight']
    rows += [dict(path=row['path'], bytes=row['bytes'], sha256=row['sha256'], kind='original_video_vae') for row in assets['files']]
    proof = args.assets_proof.resolve()
    rows.append(dict(path=str(proof), bytes=proof.stat().st_size, sha256=digest(proof), kind='assets_proof'))
    package = Path(__file__).resolve().parents[1] / 'h3_t8/freevideo_decoder'
    rows += [dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path), kind='decoder_adapter')
             for path in sorted(package.glob('*.py'))]
    rows = sorted(rows, key=lambda row: row['path'])
    verify_files(rows)
    original_config = base / 'vae/config.json'
    if digest(original_config) != digest(Path(runtime['base']) / 'vae/config.json'):
        raise ValueError('FreeVideo completed latents and this official video VAE have different configs')
    value = dict(schema=c.SCHEMA, freevideo_revision=c.FREEVIDEO_REVISION, vdn_revision=c.VDN_REVISION,
        vae_revision=c.VAE_REVISION, normalization=c.NORMALIZATION, python=runtime['python'],
        source_root=runtime['source_root'], vdn_root=runtime['vdn_root'], base=str(base), home=str(args.home.resolve()),
        files=rows, inventory_sha256=hashlib.sha256(canonical(rows).encode()).hexdigest(),
        vae_config_sha256=digest(original_config), assets_proof_sha256=digest(proof),
        pixel_mean=list(c.PIXEL_MEAN), pixel_std=list(c.PIXEL_STD),
        parameter_policy='original_FP32_streamed_no_Linear_compute_cache')
    c.validate_config(value)
    args.home.mkdir(parents=True, exist_ok=False)
    write_json_new(output, value)
    print(canonical(dict(status='decoder_config_prepared_not_GPU_qualified', config=str(output),
        sha256=digest(output), borrowed_files=len(rows), old_runtime_modified=False, compiled_by_default=False)))


if __name__ == '__main__':
    main()
