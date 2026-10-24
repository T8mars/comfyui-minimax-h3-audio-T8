"""Limited old schema review, never an exact M0/source/runtime certificate.

Keep the original snapshot unchanged. Review only type-exact, explicitly listed
metadata/expanded-range changes and ordered installed-file menu additions.
All defaults, old sockets, other metadata and old menu values remain exact.
The one named D/HF contract reviews only its two exact appended HIGH sockets.
File presence is menu eligibility evidence, not weight-content certification.
"""
from copy import deepcopy
import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath

import folder_paths

from tools.diagnose_modular_menu_drift import (
    MENU_CATEGORIES, NODE_MENU_CATEGORIES, _is_ordered_subsequence,
    installed_asset, installed_roots,
)


ABSENT = object()
EXTRA_MENU_CATEGORIES = {
    ('MiniMaxH3LearnedLatentUpscaleT8Advanced', 'model_name'): 'latent_upscale_models',
    # These exact schema owners call get_filename_list('latent_upscale_models')
    # directly or share the explicit _upscaler_options() helper. Never treat
    # generic model_name/upscaler_model or a static COMBO as file-backed.
    ('MiniMaxH3ChunkedTwoPassPlanT8Advanced', 'model_name'): 'latent_upscale_models',
    ('MiniMaxH3ChunkedTwoPassGlobalNoisePlanT8Advanced', 'model_name'): 'latent_upscale_models',
    ('MiniMaxH3ChunkedTwoPassLowSigmaPlanT8Advanced', 'model_name'): 'latent_upscale_models',
    ('MiniMaxH3ChunkedTwoPassMaskedLowSigmaPlanT8Advanced', 'model_name'): 'latent_upscale_models',
    ('MiniMaxH3ProgressiveSamplerEXPT8', 'upscaler_model'): 'latent_upscale_models',
    ('MiniMaxH3DualModelLongVideoEXPT8', 'upscaler_model'): 'latent_upscale_models',
    ('MiniMaxH3ProgressiveLongVideoEXPT8', 'upscaler_model'): 'latent_upscale_models',
    ('MiniMaxH3FastH3V2DualModelLongVideoEXPT8', 'upscaler_model'): 'latent_upscale_models',
    ('MiniMaxH3AvatarProgressiveEXPT8', 'upscaler_model'): 'latent_upscale_models',
    ('MiniMaxH3HyperFlowLongVideoEXPT8', 'upscaler_model'): 'latent_upscale_models',
}
FILE_DEFAULT_REVIEWS = {
    ('MiniMaxH3RavenGuardedLoaderT8Advanced', 'unet_name'): ('minimax', 'h3'),
    ('MiniMaxH3RavenGuardedLoaderT8Advanced', 'lora_name'): ('raven',),
}
HIGH_CONDITIONING_REVIEW = 'MiniMaxH3HyperFlowSplitT8Advanced'
SCALAR_REVIEWS = {
    'MiniMaxH3SemanticBridgeConfigT8': [
        (('description',),
         '原版通用语义 / BUNNY动作语义。默认0.10；参考音频/唱歌可能退化。模型下载：https://huggingface.co/t8star/Semantic-Bridge-Comfy 。放入 models/semantic_bridge；不会自动下载或加载教师模型。',
         '手动参数，旧默认0.10不变。新手请用‘自动模型参数’节点按文件内容读取推荐值。动漫战斗固定1.0/per_token/all_tokens/chunk=0；武术v1不是1.0。固定契约不匹配会在预检报错。多桥用显式‘组合’，不要串多个Apply。'),
        (('input', 'required', 'chunk_tokens', 1, 'min'), 1, 0),
        (('input', 'required', 'chunk_tokens', 1, 'tooltip'), ABSENT,
         '0 = whole sequence; required by some trained Transformer bridges.'),
        (('input', 'required', 'model_name', 1, 'tooltip'),
         '原版FP16封装／BUNNY FP32封装： https://huggingface.co/t8star/Semantic-Bridge-Comfy 。保留 models/semantic_bridge/t8_compat 子目录；不是LoRA或H3主模型。',
         '原版/BUNNY：https://huggingface.co/t8star/Semantic-Bridge-Comfy ；T8动漫战斗：https://huggingface.co/t8star/semantic_bridge_T8-comic-combat 。放在 models/semantic_bridge/t8_compat；不是LoRA或H3主模型。'),
    ],
    'MiniMaxH3StageSaveEXPT8': [
        (('description',),
         'Writes a new immutable native AV artifact under output/MiniMaxH3/stage_artifacts. Outputs exact path and mandatory SHA for explicit restoration. Existing artifacts are never overwritten. Unknown executable stacks can be archived but need an identity adapter before certified persistent reuse; inspect report_json.',
         'Writes a new immutable native AV artifact under output/MiniMaxH3/stage_artifacts. Outputs exact path and mandatory SHA for explicit restoration. Existing artifacts are never overwritten. Incomplete or unknown sampler execution is rejected before a completed artifact is written; inspect report_json.'),
    ],
}


def parent(value, path):
    for key in path[:-1]:
        value = value[key]
    return value


def same(first, second):
    return type(first) is type(second) and first == second


def _reviewed_menu_asset(roots, name):
    """Logical model roots may contain explicit user symlinks, just like Core.

    This proves current file presence only, not model compatibility or content.
    It never follows a path expression supplied outside an actual menu name.
    """
    normalized = name.replace('\\', '/')
    path = PurePosixPath(normalized)
    assert (normalized and not path.is_absolute() and not PureWindowsPath(name).is_absolute()
            and ':' not in normalized and '\x00' not in normalized
            and all(part not in ('', '.', '..') for part in normalized.split('/'))), name
    for root in roots:
        candidate = root.joinpath(*path.parts)
        if candidate.is_file():
            return candidate.resolve(strict=True)
    return None


def assert_reviewed_legacy_schemas(entries, live_info, models_dir=None, *, review_missing_files=False):
    assert len({row['id'] for row in entries}) == len(entries)
    roots = installed_roots(Path(models_dir or folder_paths.models_dir) / 'loras')
    roots['latent_upscale_models'] = ((Path(models_dir or folder_paths.models_dir) / 'latent_upscale_models').resolve(),)
    changes = []
    for entry in entries:
        node_id = entry['id']
        assert node_id in live_info
        old, new = entry['info'], deepcopy(live_info[node_id])
        if node_id == HIGH_CONDITIONING_REVIEW:
            before = old.get('input', {}).get('optional', {})
            after = new.get('input', {}).get('optional', {})
            if not same(before, after):
                old_order = old.get('input_order', {}).get('optional')
                new_order = new.get('input_order', {}).get('optional')
                assert before == {'negative': ['CONDITIONING', {}]}, (node_id, 'old_optional')
                assert after == {'negative': ['CONDITIONING', {}],
                                 'positive_high': ['CONDITIONING', {}],
                                 'negative_high': ['CONDITIONING', {}]}, (node_id, 'exact_HIGH_sockets')
                assert old_order == ['negative'], (node_id, 'old_optional_order')
                assert new_order == ['negative', 'positive_high', 'negative_high'], (node_id, 'HIGH_append_order')
                new['input']['optional'] = deepcopy(before)
                new['input_order']['optional'] = list(old_order)
                changes.append({'id': node_id, 'review': 'exact_D_HIGH_conditioning_append',
                                'optional_appended': ['positive_high', 'negative_high']})
        for path, before, after in SCALAR_REVIEWS.get(node_id, ()):
            old_parent, new_parent = parent(old, path), parent(new, path)
            key = path[-1]
            old_value, new_value = old_parent.get(key, ABSENT), new_parent.get(key, ABSENT)
            if same(old_value, new_value):
                continue
            assert same(old_value, before) and same(new_value, after), (node_id, path)
            if before is ABSENT:
                del new_parent[key]
            else:
                new_parent[key] = before
            changes.append({'id': node_id, 'path': list(path), 'review': 'exact_named_delta'})
        old_fields = old.get('input', {}).get('required', {})
        new_fields = new.get('input', {}).get('required', {})
        for field in old_fields.keys() & new_fields.keys():
            category = (NODE_MENU_CATEGORIES.get((node_id, field))
                        or EXTRA_MENU_CATEGORIES.get((node_id, field)) or MENU_CATEGORIES.get(field))
            if category is None:
                continue
            before, after = old_fields[field], new_fields[field]
            if before == after:
                continue
            old_options, new_options = before[1].get('options'), after[1].get('options')
            assert type(old_options) is list and type(new_options) is list
            assert all(type(name) is str for name in old_options + new_options)
            assert len(set(old_options)) == len(old_options) and len(set(new_options)) == len(new_options)
            removed = [name for name in old_options if name not in new_options]
            if review_missing_files:
                for name in removed:
                    # A sentinel/static enum cannot disappear just because it
                    # has no file. Only recognized Core model-file formats.
                    assert PurePosixPath(name.replace('\\', '/')).suffix.lower() in folder_paths.supported_pt_extensions
                    assert _reviewed_menu_asset(roots[category], name) is None, (node_id, field, name)
                retained = [name for name in old_options if name not in removed]
                assert _is_ordered_subsequence(retained, new_options), (node_id, field)
            else:
                assert _is_ordered_subsequence(old_options, new_options), (node_id, field)
            additions = [name for name in new_options if name not in old_options]
            for name in additions:
                asset = (_reviewed_menu_asset(roots[category], name) if review_missing_files
                         else installed_asset(roots[category], name))
                assert asset is not None and asset.stat().st_size > 0
            default_reviewed = False
            if review_missing_files and before[1].get('default') != after[1].get('default'):
                tokens = FILE_DEFAULT_REVIEWS.get((node_id, field))
                assert tokens is not None and before[1].get('default') in removed, (node_id, field, 'default')
                # The actual old source is byte-identical. Only its documented
                # filename-dependent selection may change with the filesystem.
                assert entry.get('source') == 'h3_t8/nodes_raven_streaming_advanced.py'
                source = Path(__file__).resolve().parents[1] / entry['source']
                assert hashlib.sha256(source.read_bytes()).hexdigest() == entry.get('source_sha256')
                expected = next((name for name in new_options if all(token in name.casefold() for token in tokens)),
                                new_options[0] if new_options else None)
                assert after[1].get('default') == expected, (node_id, field, 'computed_default')
                after[1]['default'] = before[1]['default']
                default_reviewed = True
            after[1]['options'] = old_options
            change = {'id': node_id, 'field': field, 'category': category, 'added': additions,
                      'review': 'ordered_existing_file_additions_not_model_certificate'}
            if review_missing_files:
                change.update(removed_absent_files=removed,
                              review='explicit_filesystem_drift_not_schema_or_model_certificate')
                if default_reviewed:
                    change['default_from_unchanged_source_after_file_removal'] = True
            changes.append(change)
        assert new == old, node_id
    return changes
