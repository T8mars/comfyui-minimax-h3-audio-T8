"""Limited old schema review, never an exact M0/source/runtime certificate.

Keep the original snapshot unchanged. Review only type-exact, explicitly listed
metadata/expanded-range changes and ordered installed-file menu additions.
All defaults, sockets, other metadata and old menu values remain exact.
File presence is menu eligibility evidence, not weight-content certification.
"""
from copy import deepcopy
from pathlib import Path

import folder_paths

from tools.diagnose_modular_menu_drift import (
    MENU_CATEGORIES, NODE_MENU_CATEGORIES, _is_ordered_subsequence,
    installed_asset, installed_roots,
)


ABSENT = object()
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


def assert_reviewed_legacy_schemas(entries, live_info, models_dir=None):
    assert len({row['id'] for row in entries}) == len(entries)
    roots = installed_roots(Path(models_dir or folder_paths.models_dir) / 'loras')
    changes = []
    for entry in entries:
        node_id = entry['id']
        assert node_id in live_info
        old, new = entry['info'], deepcopy(live_info[node_id])
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
            category = NODE_MENU_CATEGORIES.get((node_id, field)) or MENU_CATEGORIES.get(field)
            if category is None:
                continue
            before, after = old_fields[field], new_fields[field]
            if before == after:
                continue
            old_options, new_options = before[1].get('options'), after[1].get('options')
            assert type(old_options) is list and type(new_options) is list
            assert all(type(name) is str for name in old_options + new_options)
            assert len(set(old_options)) == len(old_options) and len(set(new_options)) == len(new_options)
            assert _is_ordered_subsequence(old_options, new_options), (node_id, field)
            additions = [name for name in new_options if name not in old_options]
            for name in additions:
                asset = installed_asset(roots[category], name)
                assert asset.stat().st_size > 0
            after[1]['options'] = old_options
            changes.append({'id': node_id, 'field': field, 'category': category,
                            'added': additions, 'review': 'ordered_existing_file_additions_not_model_certificate'})
        assert new == old, node_id
    return changes
