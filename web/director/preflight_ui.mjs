// Fast, draft-only hints. Never load models, read media bytes, or authorize queueing.
// The server compiler remains authoritative for aliases, dimensions and dependencies.
export function directorDraftIssues(doc, shot, assets = []) {
    if (!shot) return [];
    const issues = [], media = new Map(assets.map(asset => [asset.id, asset]));
    const add = (field, message) => issues.push({shot_id:shot.id, field, message});
    const shared = doc.sharedRefs || [], refs = [...new Set([...shared, ...(shot.mode === 'text' ? [] : shot.refs || [])])];
    const first = ['first','ends'].includes(shot.mode) ? shot.first : null;
    const last = shot.mode === 'ends' ? shot.last : null;
    if (shot.mode === 'first' && !first) add('source','请添加首帧。');
    if (shot.mode === 'ends' && !last) add('source','请添加尾帧；仅尾帧模式可以不设首帧。');
    if (shot.mode === 'refs' && !refs.length && shot.sound !== 'voice') add('source','请分配参考素材，或添加全片共享参考。');
    const active = [...refs, first, last, ...(shot.sound === 'native' ? [] : [shot.audio])].filter(Boolean);
    if (active.some(id => !media.has(id) || media.get(id).missing)) add('source','生效素材未就绪，请检查上传或重连。');
    for (const [id, label] of [[first,'首帧'],[last,'尾帧']]) {
        if (id && media.has(id) && media.get(id).kind !== 'image') add('source',label+'需要图片素材。');
    }
    if (shot.sound !== 'native') {
        const audio = media.get(shot.audio);
        if (audio?.kind !== 'audio') add('audio','请选择本镜使用的音频素材。');
        else if (!Number.isFinite(shot.start) || !Number.isFinite(shot.end) || shot.start < 0 || shot.end <= shot.start || shot.end > audio.duration + 0.001) add('audio','音频选区无效，请检查起止时间。');
    }
    if (shot.sound === 'record' && (!first || last || refs.some(id => id !== first && id !== shot.audio))) add('source','原音驱动需一张首帧，不叠加尾帧或其他参考。');
    const advanced = shot.writingMode === 'advanced', events = advanced ? shot.events || [] : [];
    const prompt = advanced ? shot.prompt : shot.simplePrompt;
    if((doc.global||'').includes('@missing_'))add('global','全片设定包含未绑定引用，请重新选择对应素材。');
    if((prompt||'').includes('@missing_')||events.some(event=>(event.text||'').includes('@missing_')))add('prompt','本镜包含未绑定引用，请重新选择对应素材，不能直接生成。');
    if (!(prompt || '').trim() && !events.some(event => (event.text || '').trim()) && shot.sound !== 'record') add('prompt','填写本镜提示词，或在高级模式添加台词／动作。');
    let duration = shot.manualDuration ?? shot.duration;
    if (shot.autoDuration && events.length) duration = Math.max(...events.map(event => event.end));
    else if (shot.autoDuration && shot.sound === 'record' && shot.audio) duration = shot.end - shot.start;
    if (!Number.isFinite(duration) || duration <= 0) add('timing','本镜时长必须大于 0。');
    if (events.some(event => !Number.isFinite(event.start) || !Number.isFinite(event.end) || event.start < 0 || event.end <= event.start || event.end > duration || !Number.isInteger(event.start * 24) || !Number.isInteger(event.end * 24))) add('events','事件时间需要在本镜范围内，并落在 24fps 帧网格。');
    return issues;
}
