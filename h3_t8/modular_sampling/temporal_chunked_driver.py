"""Convenience chain over the same separated scoped windows; no legacy edits."""
import json

from ..temporal_dialogue_bank import validate_window_bank
from .chunked_v5 import lift_standard_joint, prepare_standard_joint
from .chunked_v5_effects import bind_v5_eav, audit_v5_eav
from .temporal_chunked_v5 import sample_scoped_v5_window
from .temporal_chunked_relay import bind_scoped_v5_relay, audit_scoped_relay


def run_scoped_v5(model, source, plan, noise, sampler, sigmas, bank, *,
                    eav_config=None, relay_config=None, clip=None, negative=None, cfg=1.0):
    validate_window_bank(bank, source, plan)
    if bank.executor_contract != 'v5_joint_refine4':
        raise ValueError('Scoped integrated PASS2 requires v5 standard joint refine4')
    lifted, receipt, lift_report = lift_standard_joint(source, plan)
    prepared, prepare_report = prepare_standard_joint(source, lifted, receipt, plan, noise)
    previous, reports, effect_reports = None, [], []
    for index in range(len(bank.encoded)):
        selected, positive = model, bank.encoded[index].positive
        relay_runtime = eav_runtime = None
        if relay_config is not None:
            selected, positive, relay_runtime, _ = bind_scoped_v5_relay(
                selected, clip, source, lifted, prepared, plan, sigmas, bank, index, relay_config, previous)
        if eav_config is not None:
            selected, eav_runtime, _, _ = bind_v5_eav(selected, sigmas, source, lifted,
                prepared, plan, index, previous.base if previous else None, eav_config,
                positive if relay_runtime and relay_runtime.runtime else None)
        output, base, current, report = sample_scoped_v5_window(selected, source, lifted,
            prepared, plan, noise, sampler, sigmas, bank, index, previous, positive, negative, cfg)
        audits = {}
        if relay_runtime is not None and (eav_runtime is None or relay_runtime.runtime is None):
            audits['relay'] = json.loads(audit_scoped_relay(current, source, plan, prepared, bank, relay_runtime)[1])
        if eav_runtime is not None:
            audits['eav'] = json.loads(audit_v5_eav(base, prepared, eav_runtime)[1])
        reports.append(json.loads(report))
        effect_reports.append(audits)
        previous = current
    report = dict(status='scoped_joint_av_chain_completed_quality_unverified',
                  bank_sha256=bank.sha256, windows=reports, effects=effect_reports,
                  lift=json.loads(lift_report), preparation=json.loads(prepare_report),
                  audio_output='refined_joint_audio', hard_time_isolation=False, quality_accepted=False)
    return output, base, current, lifted, prepared, json.dumps(report, ensure_ascii=False, sort_keys=True)
