// Opt-in candidate inputs only; histories and unadopted drafts are not execution.
export function radarCandidateInputState(doc, shot, assets) {
    const candidate = shot.promptCandidate;
    if (!candidate?.active) return undefined;
    const media = new Map(assets.map(row => [row.id, row.sha256 ?? null]));
    const evidence = candidate.options.facts ? shot.sourceEvidence ?? null : null;
    const intent = candidate.options.intent ? shot.creativeIntent ?? null : null;
    const packetMedia = packet => [packet.source, ...(packet.audio_source ? [packet.audio_source] : []),
        ...packet.claims.filter(row => row.frame).map(row => row.frame)];
    const status = other => {
        if (!other?.sourceEvidence) return other ? 'legacy' : 'missing';
        const row = other.sourceEvidence;
        if (packetMedia(row.packet).some(source => media.get(source.asset_id) !== source.sha256)) return 'source_changed';
        return !row.review ? 'unreviewed' : row.review.packet_sha256 === row.packet.sha256 ? 'confirmed' : 'review_stale';
    };
    const referenced = new Set([...(doc.sharedRefs || []), ...(shot.tray || []), ...(shot.refs || []),
        ...['first', 'last', 'audio'].map(key => shot[key]).filter(Boolean)]);
    if (evidence) for (const source of packetMedia(evidence.packet)) referenced.add(source.asset_id);
    const disabled = new Set(shot.skillDisabled || []);
    const bindings = [...(shot.skillInherit === false ? [] : doc.sharedSkills || []), ...(shot.skillBindings || [])]
        .filter(row => row.enabled && !disabled.has(row.id));
    return {
        candidate: Object.fromEntries(Object.entries(candidate).filter(([key]) => key !== 'active')),
        inputs: {
            schema: 't8.director.prompt_inputs.v1', global: doc.global || '',
            draft: Object.fromEntries(['writingMode', 'simplePrompt', 'prompt', 'events', 'first', 'last', 'audio', 'mode', 'sound']
                .map(key => [key, shot[key] ?? null])),
            references: {shared: doc.sharedRefs || [], tray: shot.tray || [], refs: shot.refs || []},
            assets: Object.fromEntries([...referenced].sort().map(id => [id, media.get(id) ?? null])),
            evidence, intent,
            dependencies: (intent?.dependencies || []).map(dep => {
                const other = doc.shots?.find(row => row.id === dep.shot_id);
                return {declared: dep, actual_packet_sha256: other?.sourceEvidence?.packet.sha256 ?? null,
                    review: other?.sourceEvidence?.review ?? null, status: status(other)};
            }), bindings, options: candidate.options,
        },
    };
}
