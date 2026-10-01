// Persistent D1 services; UI drafts are separate from saved server revisions.
export function directorUUID() {
    if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
    // getRandomValues is available on ordinary LAN HTTP, unlike randomUUID.
    const bytes = new Uint8Array(16);
    globalThis.crypto.getRandomValues(bytes);
    bytes[6] = (bytes[6] & 15) | 64;
    bytes[8] = (bytes[8] & 63) | 128;
    const hex = [...bytes].map(b => b.toString(16).padStart(2, "0"));
    return [hex.slice(0,4), hex.slice(4,6), hex.slice(6,8), hex.slice(8,10), hex.slice(10)].map(a=>a.join("")).join("-");
}
export function directorSeed(shot, fallback = 26091901) {
    const raw = shot?.seed ?? fallback;
    if (typeof raw === "boolean" || raw === "" || raw == null || !/^[0-9]+$/.test(String(raw))) throw Error("种子必须是非负整数");
    const seed = Number(raw);
    if (!Number.isSafeInteger(seed) || seed < 0) throw Error("种子超出可精确保存的整数范围");
    return seed;
}
export function newDirectorSeed(previous, random = globalThis.crypto) {
    const value = random.getRandomValues(new Uint32Array(1))[0];
    return value === previous ? (value + 1) % 4294967296 : value;
}
export function directorOutputVideos(data) {
    const result = [];
    for (const value of Object.values(data?.outputs || {})) {
        for (const key of ["videos", "gifs", "video"]) {
            if (Array.isArray(value?.[key])) result.push(...value[key]);
        }
        // SafeAVSave reports MP4 under Core's legacy `images` key.
        if (Array.isArray(value?.images)) result.push(...value.images.filter(item => /\.(mp4|mov|mkv|webm)$/i.test(item?.filename || "")));
    }
    return result;
}
export function directorResultKey(record) {
    if (record?.prompt_id) return record.prompt_id;
    const media = directorOutputVideos(record)[0];
    return media ? 'legacy:' + JSON.stringify([record.shot_id, media.type || 'output', (media.subfolder || '').replaceAll('\\', '/'), media.filename]) : null;
}
export function mergeDirectorResults(previous, incoming) {
    const records = new Map();
    for (const item of [...previous, ...incoming]) {
        const key = directorResultKey(item);
        if (!key) continue;
        const old = records.get(key);
        // A delayed pending index must not replace a completed result.
        if (old?.state === 'success' && directorOutputVideos(old).length && !directorOutputVideos(item).length && !item.integrity_error) continue;
        records.set(key, {...old, ...item});
    }
    return [...records.values()].sort((a,b) => (Number(b.submitted_at) || 0) - (Number(a.submitted_at) || 0));
}
export function selectDirectorResult(records, selectedId = null) {
    if (selectedId) return records.find(item => directorResultKey(item) === selectedId);
    return records.find(item => item.state === 'success' && directorOutputVideos(item).length) || records[0];
}
export function directorTaskLabel(state) {
    return {not_submitted:'待生成',pending:'已提交',queued:'排队中',running:'生成中',success:'已完成',error:'失败',cancelled:'已取消',unknown:'状态待核对',needs_review:'回执或媒体需核对',invalid:'输出需核对',missing:'文件缺失',corrupt:'文件损坏'}[state] || '状态待核对';
}
// Access to the storage property itself can throw (privacy modes / sandboxed frames).
const storageRead = (kind, key) => { try { return globalThis[kind].getItem(key); } catch { return null; } };
const storageWrite = (kind, key, value) => { try { globalThis[kind].setItem(key, value); return true; } catch { return false; } };
const storageRemove = (kind, key) => { try { globalThis[kind].removeItem(key); } catch { /* Optional recovery cache. */ } };
const clearAdoption = doc => { for (const shot of doc.shots) { delete shot.adoptedResultId; delete shot.filmTrim; } };
const stableJSON = value => JSON.stringify(value, function(_key, item) { return item && typeof item === 'object' && !Array.isArray(item) ? Object.fromEntries(Object.keys(item).sort().map(key=>[key,item[key]])) : item; });
export function directorShotInputKey(doc, shot, assets = []) {
    const sampling = shot.samplingInherit === false ? shot.sampling : doc.sampling;
    const mode = sampling?.mode || 'single';
    const activeSampling = Object.fromEntries(Object.entries(sampling || {mode:'single'}).filter(([key])=>!['single','two_pass','hyperflow'].includes(key)));
    if(mode==='hyperflow' && sampling.variant==='single8')delete activeSampling.high_loras;
    const generation = {...doc.generation};
    if(mode!=='single')for(const key of ['resolution_mp','lora','loras','lora_mode','lora_strength'])delete generation[key];
    else if(sampling){
        if(sampling.resolution_mp!=null)delete generation.resolution_mp;
        if(sampling.loras!=null)for(const key of ['lora','loras','lora_mode','lora_strength'])delete generation[key];
    }
    // Row IDs/source labels and disabled drafts are editing metadata, not sampling inputs.
    for(const holder of [activeSampling,generation])for(const key of ['loras','low_loras','high_loras']){
        if(Array.isArray(holder[key]))holder[key]=holder[key].filter(row=>row.enabled!==false).map(row=>({name:row.name,strength:row.strength??1}));
    }
    if(activeSampling.lora_mode==='none')activeSampling.loras=[];
    if(generation.lora_mode==='none')generation.loras=[];
    const ids = [...new Set([...(doc.sharedRefs || []),...(shot.tray || []),shot.first,shot.last,...(shot.refs || []),shot.audio].filter(Boolean))];
    const media = new Map(assets.map(item=>[item.id,item]));
    const value = {
        global:doc.global || '', sharedRefs:doc.sharedRefs || [], mode:shot.mode, sound:shot.sound,
        first:shot.mode==='first'||shot.mode==='ends'?shot.first:null,last:shot.mode==='ends'?shot.last:null,
        refs:shot.mode==='text'?[]:shot.refs || [],tray:shot.tray || [],
        audio:shot.sound==='native'?null:{id:shot.audio,start:shot.start,end:shot.end},
        writingMode:shot.writingMode || 'simple',
        prompt:shot.writingMode==='advanced'?shot.prompt || '':shot.simplePrompt || '',
        events:shot.writingMode==='advanced'?(shot.events || []).map(({start,end,text})=>({start,end,text})):[],
        duration:shot.duration,manualDuration:shot.manualDuration,autoDuration:shot.autoDuration,
        ratio:doc.sharedRatio?doc.ratio:shot.ownRatio || shot.ratio,
        seed:directorSeed(shot), sampling:activeSampling,generation,
        d3:shot.d3Inherit===false?shot.d3:doc.d3 || shot.d3,
        assets:ids.map(id=>{const a=media.get(id)||{};return {id,sha256:a.sha256,kind:a.kind,width:a.width,height:a.height,duration:a.duration,has_audio:a.has_audio};}),
    };
    const sorted=v=>Array.isArray(v)?v.map(sorted):v&&typeof v==='object'?Object.fromEntries(Object.keys(v).sort().map(key=>[key,sorted(v[key])])):v;
    return JSON.stringify(sorted(value));
}
export function moveDirectorShot(shots, id, target) {
    const from=shots.findIndex(shot=>shot.id===id);
    if(from<0||!Number.isInteger(target)||target<0||target>=shots.length||from===target)return shots;
    const next=[...shots];next.splice(target,0,next.splice(from,1)[0]);return next;
}
export function makeDirectorServices(ctx) {
    const { $, esc, notify, showDialog } = ctx;
    // The HTML prototype is intentionally kept as the stable D1 layout.  At
    // runtime the real service owns the badge/copy so users cannot mistake the
    // current D2a–D2c queue bridge for the old preflight-only page.
    const badge = $(".o-badge");
    if (badge) badge.textContent = "D2a–D2c · 真实生成";
    const note = $(".o-note");
    if (note) note.textContent = "D2a–D2c：文字、首帧、首尾、参考素材、原音驱动和参考音色均按正式 Core 配方编译；GPU画质仍需逐项验收。";
    const generationButton = $("[data-action=\"generate\"]");
    if (generationButton) {
        const normalizeLabel = () => {
            if (/编译预检|生成第/.test(generationButton.textContent || "")) {
                generationButton.textContent = "生成当前镜头";
            }
        };
        new MutationObserver(normalizeLabel).observe(generationButton, { childList: true, characterData: true, subtree: true });
        normalizeLabel();
    }
    const base = new URL(".", import.meta.url);
    const id = directorUUID;
    const onlineNote = "D2a–D2c：文字、首帧、首尾、参考素材、原音驱动和参考音色均按正式 Core 配方编译；GPU画质仍需逐项验收。";
    let tab = storageRead('sessionStorage', 't8director.tab') || id();
    storageWrite('sessionStorage', 't8director.tab', tab);
    let draftKey = "t8director.draft:" + tab;
    let projectId = id(), revision = 0, title = "我的第一部短片", busy = false, saving = false, reconnectId = null;
    let projectEpoch = 0;
    let resultRequest = 0, compileRequest = 0, d3Request = 0, projectLoadRequest = 0, editEpoch = 0;
    let reconnectTarget = null, lastJobDialog = null;
    let activeBatch = null, pauseBatch = false, versionRequest = 0, tasksRequest = 0, versionCopy = null;
    const completedResults = new Map();
    const contextToken = () => `${projectId}:${projectEpoch}`;
    let latest = null, latestSnapshot = null, importFile = null, activeJobId = null, activeUpload = null, stopWatchingJob = null;
    let activeJobKey = "t8director.activeJob:" + tab;
    let scopeFallbackDraft = null, releaseScope = null, scopeClaimed = false;
    const pendingRequests = new Map();
    let pendingRequestReview = null;
    let pendingProjectSwitch = null;
    $("[data-dialog]").addEventListener('close',()=>{if(!$("[data-dialog]").open)pendingProjectSwitch=null;});
    // sessionStorage is copied by window.open. Hold an exclusive same-origin
    // lock for this document so a second live document cannot share its draft.
    async function claimScope(recoverCurrent = false) {
        const oldTab = tab;
        const recovery = recoverCurrent ? JSON.stringify(envelope()) : storageRead('localStorage', draftKey);
        const job = storageRead('sessionStorage', activeJobKey);
        const editorDrafts = storageRead('localStorage', 't8director.editors:' + oldTab);
        const locks = window.navigator?.locks;
        if (locks?.request) {
            const acquire = candidate => new Promise((resolve, reject) => {
                locks.request('t8director.tab:' + candidate, {ifAvailable:true}, async lock => {
                    if (!lock) { resolve(false); return; }
                    await new Promise(release => { releaseScope = release; resolve(true); });
                }).catch(reject);
            });
            try { while (!await acquire(tab)) tab = id(); }
            catch { tab = id(); } // No lock service: use a fresh isolated namespace.
        } else if (window.navigator) {
            // Older/insecure browsers: transfer the recovery content to a new
            // document namespace; never discard it when refreshing.
            tab = id();
        }
        draftKey = 't8director.draft:' + tab;
        activeJobKey = 't8director.activeJob:' + tab;
        storageWrite('sessionStorage', 't8director.tab', tab);
        if (tab !== oldTab) {
            if (recovery) { scopeFallbackDraft = recovery; storageWrite('localStorage', draftKey, recovery); }
            if (job) storageWrite('sessionStorage', activeJobKey, job);
            if (editorDrafts) storageWrite('localStorage', 't8director.editors:' + tab, editorDrafts);
        }
        scopeClaimed = true;
    }
    ctx.root.inert = true;
    const scopeReady = claimScope();
    window.addEventListener('pagehide', () => { releaseScope?.(); releaseScope = null; });
    window.addEventListener('pageshow', event => {
        if (!event.persisted) return;
        ctx.root.inert = true;
        claimScope(true).then(()=>draft()).finally(()=>{ctx.root.inert=false;});
    });
    const batchKey = owner => "t8director.batch:" + owner;
    const savedBatchId = owner => { try { return localStorage.getItem(batchKey(owner)); } catch { return null; } };
    const saveBatchId = (owner, batchId) => { try { localStorage.setItem(batchKey(owner), batchId); } catch { notify("浏览器无法记住批次链接；请保存此批次 ID：" + batchId); } };
    const assetURL = aid => new URL("assets/" + aid, base).href;
    const connectionNotice = () => {
        const online = navigator.onLine !== false;
        const badge = $(".o-badge");
        if (badge && !activeJobId) badge.textContent = online ? "D2a–D2c · 真实生成" : "离线 · 草稿仍保留";
        const note = $(".o-note");
        if (note) note.textContent = online ? onlineNote : "当前浏览器离线：保存／上传／生成需恢复连接后手动重试，不会自动补交；请确认草稿保护状态，必要时导出备份。";
    };
    window.addEventListener("online", connectionNotice);
    window.addEventListener("offline", connectionNotice);
    connectionNotice();
    function uploadNotice(message, cancellable = false) {
        notify(message);
        const notice = $("[data-notice]");
        if (!notice) return;
        notice.querySelector("[data-upload-cancel]")?.remove();
        if (!cancellable) return;
        const button = document.createElement("button");
        button.type = "button";
        button.dataset.service = "cancel-upload";
        button.dataset.uploadCancel = "";
        button.textContent = "取消上传";
        button.title = "取消网络上传；未确认的素材不会加入本镜";
        notice.append(" ", button);
    }
    function rememberJob(promptId, recipe, shotId, owner) {
        try { sessionStorage.setItem(activeJobKey, JSON.stringify({ prompt_id: promptId, recipe: recipe || "director", project_id: owner, shot_id: shotId, started_at: Date.now() })); }
        catch { /* Storage can be disabled; in-memory polling still remains safe. */ }
    }
    function pendingJob() {
        try {
            const value = JSON.parse(sessionStorage.getItem(activeJobKey) || "null");
            return value && typeof value.prompt_id === "string" ? value : null;
        } catch { return null; }
    }
    function showJobDialog(promptId, heading, body, force = false) {
        const dialog = $("[data-dialog]");
        // Background task updates must not replace an editor, library or confirmation.
        if (!force && (otherModalOpen() || (dialog.open && !dialog.dataset.jobId))) return;
        showDialog(heading, body);
        dialog.dataset.jobId = promptId;
    }
    function otherModalOpen() {
        return [...(ctx.root.querySelectorAll?.('dialog[open]') || [])].some(dialog=>dialog!==$('[data-dialog]'));
    }
    function showActiveJob(force = true) {
        if (!activeJobId) {
            if (lastJobDialog) showJobDialog(...lastJobDialog, force);
            return;
        }
        const owner = pendingJob()?.project_id;
        showJobDialog(activeJobId, "导演台任务进行中", `<p>任务 ID：${esc(activeJobId)}</p><p>所属项目：${esc(owner || "旧记录未标注")}${owner && owner !== projectId ? "（不是当前项目）" : ""}</p><p data-job-state>正在跟踪原任务，不会重复提交。</p><button data-service="cancel-job">取消此任务</button>`, force);
    }
    function forgetJob(promptId = activeJobId) {
        try {
            const value = pendingJob();
            if (!promptId || !value || value.prompt_id === promptId) sessionStorage.removeItem(activeJobKey);
        } catch { /* Best effort only; the Core prompt remains the source of truth. */ }
    }
    const materialize = a => ({ ...a, url: assetURL(a.id), file: { arrayBuffer: async () => {
        const response = await fetch(assetURL(a.id));
        if (!response.ok) throw Error("服务端素材缺失，请重新上传并重连");
        return response.arrayBuffer();
    } } });
    const envelope = (selectedCurrent = ctx.current()) => {
        const aliasMaps = {};
        for (const s of ctx.doc().shots) aliasMaps[s.id] = Object.fromEntries([...ctx.tokenMap(s)].map(([aid, alias]) => [alias, aid]));
        return { schema: "t8.minimax_h3.director_project", version: 2, id: projectId, revision, title,
            doc: structuredClone(ctx.doc()), current: selectedCurrent, aliasMaps,
            assets: [...ctx.assets().values()].map(a => Object.fromEntries(Object.entries(a).filter(([key]) => !["url", "file", "peaks", "missing"].includes(key)))) };
    };
    async function request(path, body, method = body ? "POST" : "GET", signal) {
        const response = await fetch(new URL(path, base), { method, credentials: "same-origin", signal,
            ...(body ? { headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {}) });
        const data = await response.json();
        if (!response.ok) { const e = Error(data.error || "服务端请求失败"); e.status = response.status; e.data = data; throw e; }
        return data;
    }
    async function loadResults(owner = projectId, autoOpen = false) {
        const epoch = projectEpoch, sequence = ++resultRequest;
        const shots = ctx.doc().shots.map(shot => ["shot", shot.id]);
        const query = new URLSearchParams(shots).toString();
        const data = await request("results/" + encodeURIComponent(owner) + (query ? "?" + query : ""));
        if (owner === projectId && epoch === projectEpoch && sequence === resultRequest) {
            // A lagging server index cannot erase a completion already observed here.
            for (const item of data.results || []) {
                if (completedResults.get(item.shot_id)?.prompt_id === item.prompt_id && directorOutputVideos(item).length) completedResults.delete(item.shot_id);
            }
            const observed = [...completedResults.values()];
            ctx.setResults(mergeDirectorResults(observed, data.results || []), autoOpen);
        }
        return data.results || [];
    }
    function draft() {
        latest = null;
        const checked = $("[data-preflight-status]");
        if (checked) checked.textContent = "草稿或镜头已更改，请重新检查；下方是旧检查记录。";
        const cached = scopeClaimed && storageWrite('localStorage', draftKey, JSON.stringify(envelope()));
        $("[data-save]").textContent = cached ? `本标签草稿已恢复保护 · 服务端版本 ${revision} · 请保存项目` : "浏览器草稿保存失败：可先导出 project.json，再重试服务端保存";
        return !!cached;
    }
    function syncProjectURL() {
        const url = new URL(location.href);
        url.searchParams.set("project_id", projectId);
        window.history.replaceState(window.history.state, "", url.href);
    }
    function beginProjectLoad() {
        return { sequence: ++projectLoadRequest, context: contextToken(), snapshot: JSON.stringify(envelope()), edits: editEpoch };
    }
    function acceptProjectLoad(ticket) {
        if (ticket.sequence !== projectLoadRequest || ticket.context !== contextToken()) return false;
        if (ticket.edits !== editEpoch || ticket.snapshot !== JSON.stringify(envelope())) {
            notify("载入期间有新编辑，已保留当前草稿；请保存后重新打开或导入项目。");
            return false;
        }
        return true;
    }
    function editorRecovery(key, value) {
        const storageKey = 't8director.editors:' + tab;
        let entries; try { entries = JSON.parse(storageRead('localStorage', storageKey) || '{}'); } catch { entries = {}; }
        if (!entries || typeof entries !== 'object' || Array.isArray(entries)) entries = {};
        const scopedKey = projectId + ':' + key;
        if (value === undefined) {
            const saved=entries[scopedKey];
            return saved && typeof saved.text==='string' && Array.isArray(saved.shared) &&
                Array.isArray(saved.tokens) && saved.tokens.every(pair=>Array.isArray(pair)&&pair.length===2&&pair.every(v=>typeof v==='string')) ? saved : null;
        }
        if (value === null) delete entries[scopedKey]; else entries[scopedKey] = value;
        return storageWrite('localStorage', storageKey, JSON.stringify(entries));
    }
    function switchProject(p, ticket, note = '') {
        if (!acceptProjectLoad(ticket)) return;
        if (!storageWrite('localStorage', draftKey + ':backup:' + projectId, ticket.snapshot)) {
            pendingProjectSwitch = {p, ticket, note, exported:false};
            showBackupRecovery();
            return;
        }
        finishProjectSwitch(p, note);
    }
    function showBackupRecovery() {
        const pending = pendingProjectSwitch;
        showDialog('浏览器无法备份当前草稿', '<p>浏览器存储已满或不可用，当前工程没有被替换。请先导出当前草稿，并确认文件已保存，再继续载入。</p>' +
            '<button data-service="backup-export">导出当前草稿备份</button> ' +
            `<button data-service="backup-continue" ${pending?.exported ? '' : 'disabled'}>已确认备份，继续载入</button> <button data-action="close">取消</button>`);
    }
    function finishProjectSwitch(p, note) {
        pendingProjectSwitch = null;
        hydrate(p); $("[data-dialog]").close(); draft();
        if (note) notify(note);
    }
    function hydrate(p) {
        if (p.schema !== "t8.minimax_h3.director_project" || ![0, 1, 2].includes(p.version) || !Array.isArray(p.doc?.shots) || !p.doc.shots.length) throw Error("不是支持的导演台项目，保留原文件");
        // Never use client-supplied URLs, filesystem paths or executable code.
        projectEpoch++;
        completedResults.clear();
        projectId = p.id; revision = p.revision; title = p.title;
        const restored = structuredClone(p.doc);
        restored.sampling ??= { mode: "single" };
        ctx.replace(restored, p.current, new Map(p.assets.map(a => [a.id, materialize(a)])));
        latest = null; ctx.resetHistory(); ctx.render();
        loadResults(projectId, true).catch(error => notify("镜头成片记录暂时无法读取：" + error.message));
        refreshBatchButton(projectId).catch(error => notify("批次读取失败；没有自动提交：" + error.message));
        $("[data-project-title]").value = title;
        $("[data-save]").textContent = `已载入项目 · 服务端版本 ${revision}`;
        storageWrite('localStorage', 't8director.lastProject', projectId);
        syncProjectURL();
    }
    function download(name, value) {
        downloadBytes(name, JSON.stringify(value, null, 2));
    }
    function downloadBytes(name, bytes) {
        const url = URL.createObjectURL(new Blob([bytes], { type: "application/json" }));
        const anchor = document.createElement("a"); anchor.href = url; anchor.download = name; anchor.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    }
    async function save(copy = false) {
        if (saving) { notify('项目正在保存，请等待本次确认；新编辑仍保留在草稿。'); return; }
        saving = true; $("[data-save]").textContent = "正在保存到服务端…";
        const epoch = projectEpoch;
        try {
            const before = envelope();
            const copyKey = 't8director.copy:' + before.id;
            let pendingCopy = null;
            try { pendingCopy = pendingRequests.get(copyKey) || JSON.parse(storageRead('sessionStorage', copyKey) || 'null'); } catch { /* Corrupt optional cache. */ }
            const p = copy ? (pendingCopy?.project && !pendingCopy.rejected ? pendingCopy.project : { ...structuredClone(before), id: pendingCopy?.project?.id || id(), revision: 0, title: before.title + " · 副本" }) : before;
            let backupOK = true;
            if (copy) {
                clearAdoption(p.doc);
                // Backup is attempted BEFORE creating a server copy. A failed
                // optional backup must not turn a successful remote save into failure.
                backupOK = storageWrite('localStorage', draftKey + ':backup:' + before.id, JSON.stringify(before));
                pendingCopy = {project:p}; pendingRequests.set(copyKey,pendingCopy);
                storageWrite('sessionStorage', copyKey, JSON.stringify(pendingCopy));
            }
            let saved;
            try { saved = await request("projects/" + p.id, { project: p, expected_revision: p.revision }); }
            catch (error) {
                if(copy && error.status===400){
                    pendingCopy={project:p,rejected:true};pendingRequests.set(copyKey,pendingCopy);
                    storageWrite('sessionStorage',copyKey,JSON.stringify(pendingCopy));
                }
                if (!copy || error.status !== 409) throw error;
                // A lost successful response is retried against the SAME target.
                // Never accept an unrelated/edited copy as this acknowledgement.
                const existing = await request('projects/' + p.id);
                if (existing.id !== p.id || existing.revision !== 1 || existing.title !== p.title || stableJSON(existing.doc) !== stableJSON(p.doc) || stableJSON(existing.aliasMaps || {}) !== stableJSON(p.aliasMaps || {}) || stableJSON((existing.assets || []).map(a=>a.id)) !== stableJSON(p.assets.map(a=>a.id))) {
                    throw Error('副本目标已存在或发生变化，请在“打开项目”核对：' + p.id + '。未另建副本，当前草稿保留。');
                }
                saved = existing;
            }
            const savedProject = { ...p, revision: saved.revision };
            if (copy) { pendingRequests.delete(copyKey); storageRemove('sessionStorage',copyKey); }
            // A response belongs to the document that initiated it, never a subsequently opened project.
            if (epoch !== projectEpoch) {
                storageWrite('localStorage', draftKey + ':saved:' + p.id, JSON.stringify(savedProject));
                notify("先前项目已保存；当前打开的项目与草稿保持不变。");
                return;
            }
            const copyDraft = {...structuredClone(before),id:p.id,revision:p.revision,title:before.title+' · 副本'}; clearAdoption(copyDraft.doc);
            const changed = JSON.stringify(envelope()) !== JSON.stringify(before) || (copy && stableJSON(copyDraft) !== stableJSON(p));
            if (copy) {
                backupOK = storageWrite('localStorage', draftKey + ':backup:' + before.id, JSON.stringify(envelope())) && backupOK;
                projectEpoch++;
                projectId = p.id;
                clearAdoption(ctx.doc());
                completedResults.clear(); ctx.clearResults?.(); ctx.resetHistory();
                if (title === before.title) title = p.title;
                $("[data-project-title]").value = title;
            }
            revision = saved.revision;
            // Keep edits made while the request was in flight, with the new CAS revision.
            const draftOK = draft();
            syncProjectURL();
            const lastOK = storageWrite('localStorage', 't8director.lastProject', projectId);
            $("[data-save]").textContent = changed ? `已保存请求时的版本 ${revision} · 后续编辑已保留在草稿，请再次保存` : `已真实保存到服务端 · 版本 ${revision}`;
            const recoveryWarning = !draftOK || !lastOK || !backupOK ? ' 服务端已保存，但本地恢复保护不完整；请导出当前草稿并保留项目链接。' : '';
            $("[data-save]").textContent += recoveryWarning;
            notify((copy ? '已另存编辑副本，不继承原工程的成片采用和裁剪；携带成片请使用工程 ZIP。' : '') + (changed ? "保存期间的新编辑已保留，尚未提交到服务端；可继续编辑或再次保存。" : "项目与服务端素材身份已保存；没有提交生成。") + recoveryWarning);
            parent.postMessage({ type: "t8-director:saved", project: savedProject }, location.origin);
            if(copy){ctx.render();loadResults(projectId).catch(error=>notify('副本成片记录读取失败：'+error.message));refreshBatchButton().catch(()=>{});}
        } catch (e) {
            if (epoch === projectEpoch) $("[data-save]").textContent = "保存未完成 · 草稿保留 · 可重试／另存副本";
            notify(e.message); // A conflict never auto-merges or replaces either draft.
        } finally { saving = false; }
    }
    async function upload(file) {
        if (!file) throw Error("没有选择素材");
        if (navigator.onLine === false) throw Error("当前离线，素材不会提交；请恢复连接后重试，草稿仍保留");
        if (Number.isFinite(file.size) && file.size > 1024 * 1024 * 1024) {
            throw Error("单素材超过1GiB，未上传；请分段或压缩后重试");
        }
        const form = new FormData(); form.append("file", file);
        uploadNotice("正在上传 " + file.name + "（0%），成功确认后才加入本镜；失败可重试。", true);
        // Fetch does not expose upload progress in browsers.  XHR is used only
        // for this one POST so a 1 GiB boundary is visible to a beginner and a
        // disconnect never leaves a half-registered asset in the project.
        const data = await new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            activeUpload = xhr;
            xhr.open("POST", new URL("assets", base), true);
            xhr.withCredentials = true;
            xhr.upload.onprogress = event => {
                if (!event.lengthComputable) return;
                uploadNotice("正在上传 " + file.name + "（" + Math.round(event.loaded / event.total * 100) + "%），成功确认后才加入本镜；失败可重试。", true);
            };
            xhr.onerror = () => reject(Error("素材上传中断，服务端未确认注册；请保持草稿并重试"));
            xhr.ontimeout = () => reject(Error("素材上传超时，服务端未确认注册；请保持草稿并重试"));
            xhr.onabort = () => reject(Error("已取消素材上传；服务端未确认注册，草稿与原素材不受影响"));
            xhr.onload = async () => {
                let value = {};
                try { value = JSON.parse(xhr.responseText || "{}"); } catch { reject(Error("素材上传返回无法读取，请重试")); return; }
                if (xhr.status < 200 || xhr.status >= 300) { reject(Error(value.error || "素材上传失败，请重试")); return; }
                resolve(value);
            };
            xhr.onloadend = () => { if (activeUpload === xhr) activeUpload = null; };
            xhr.send(form);
        });
        uploadNotice("素材已由服务端确认注册。", false);
        return materialize(data);
    }
    async function compile() {
        const p = envelope();
        const epoch = projectEpoch, sequence = ++compileRequest, snapshot = JSON.stringify(p);
        const result = await request("compile", { project: p, shot_id: p.current });
        // Changing project, shot or draft invalidates this preview, not the new context.
        if (epoch !== projectEpoch || sequence !== compileRequest || snapshot !== JSON.stringify(envelope())) return result;
        const shot = result.shots?.find(s => s.id === p.current);
        if (!shot?.canvas) throw Error("预检响应缺少请求镜头的画布信息，请重试。");
        latest = result;
        latestSnapshot = snapshot;
        ctx.refreshSummary?.();
        if(otherModalOpen()){notify('当前镜头预检已返回；请关闭当前窗口后查看，不打断编辑或审片。');return latest;}
        const inputs = shot.canvas.preprocessing.filter(p => p.state === "cpu_processed").map(p => `<figure><img style="max-height:300px;width:100%;object-fit:contain" src="${new URL(`assets/${p.asset_id}/input-preview?width=${shot.canvas.width}&height=${shot.canvas.height}`, base)}" alt="实际CPU首尾输入"><figcaption>实际首尾输入 ${shot.canvas.width}×${shot.canvas.height} · 等比缩放${p.padding_needed ? "＋黑边填充" : ""}，不拉伸、不裁人物</figcaption></figure>`).join("");
        const issues = (latest.errors || []).map((issue,index) => `<li>${esc(issue.message)} <button data-service="locate-issue" data-issue-index="${index}">去修正</button></li>`).join("");
        const warnings = (latest.warnings || []).map(issue => `<li>${esc(issue.message)}</li>`).join("");
        const geometry = Number.isInteger(shot.canvas.width) && Number.isInteger(shot.canvas.height)
            ? `<p>预检尺寸：${shot.canvas.width} × ${shot.canvas.height} · ${(shot.canvas.width * shot.canvas.height / 1000000).toFixed(3)} MP</p>` : "";
        showDialog("当前镜头编译预检", `<p data-preflight-status role="status">${latest.ready ? "项目与素材检查通过" : "请先解决当前镜头的问题"} · 不代表依赖、GPU 或画质已验收。</p>${geometry}${issues ? `<ul class="w-preflight-issues">${issues}</ul>` : ""}${warnings ? `<details open><summary>提醒（不等于禁止生成）</summary><ul>${warnings}</ul></details>` : ""}${inputs}<p>其他镜头的未完成草稿不会阻止当前镜头。</p><details><summary>技术详情与原始报告</summary><pre>${esc(JSON.stringify({ errors: latest.errors, warnings: latest.warnings, current_shot: shot, compilation_sha256: latest.compilation_sha256 }, null, 2))}</pre></details>`);
        notify(latest.ready ? "当前镜头预检通过，可生成这一镜。" : "当前镜头预检未通过，素材和草稿仍保留。");
        return latest;
    }
    function viewURL(item) {
        if (!item || !item.filename) return "";
        const url = new URL("/view", location.origin);
        url.searchParams.set("filename", item.filename);
        url.searchParams.set("type", item.type || "output");
        if (item.subfolder) url.searchParams.set("subfolder", item.subfolder);
        if (item.format) url.searchParams.set("format", item.format);
        return url.href;
    }
    async function watchJob(promptId, recipe, shotId, owner) {
        if (stopWatchingJob) stopWatchingJob();
        let timer = null;
        let unknownPolls = 0;
        let lastPollError = null;
        let resolveCompletion;
        const completion = new Promise(resolve => { resolveCompletion = resolve; });
        activeJobId = promptId;
        lastJobDialog = null;
        $("[data-service=\"job-status\"]").hidden = false;
        rememberJob(promptId, recipe, shotId, owner);
        showActiveJob(false);
        const finish = (state = "cancelled") => { if (timer) clearInterval(timer); timer = null; if (stopWatchingJob === finish) { stopWatchingJob = null; $("[data-service=\"job-status\"]").hidden = true; } resolveCompletion(state); };
        stopWatchingJob = finish;
        const poll = async () => {
            try {
                const data = await request("jobs/" + encodeURIComponent(promptId));
                if (activeJobId !== promptId) return; // Ignore an in-flight response after cancellation.
                lastPollError = null;
                const state = $("[data-job-state]");
                if (data.state === "unknown") {
                    unknownPolls += 1;
                    if (state) state.textContent = "Core 尚未返回任务状态，继续等待注册…";
                    if (unknownPolls >= 20) {
                        finish("unknown");
                        activeJobId = null;
                        forgetJob(promptId);
                        notify("Core 未保留该任务状态；不会重复提交，原草稿仍保留。 ");
                    }
                    return;
                }
                unknownPolls = 0;
                if (state && data.state !== "success" && data.state !== "error") {
                    const progress = data.progress?.fraction == null ? "" : ` · ${Math.round(data.progress.fraction * 100)}%`;
                    state.textContent = "Core 状态：" + directorTaskLabel(data.state) + progress + " · 任务 ID " + promptId;
                }
                if (data.state === "success" || data.state === "error") {
                    finish(data.state);
                    activeJobId = null;
                    forgetJob(promptId);
                    const videos = directorOutputVideos(data);
                    const belongsHere = owner === projectId && ctx.doc().shots.some(shot => shot.id === shotId);
                    if (belongsHere) {
                        const record = { project_id: owner, shot_id: shotId, prompt_id: promptId, state: data.state, outputs: data.outputs || {}, recipe, submitted_at: Date.now() / 1000 };
                        if (videos.length) completedResults.set(shotId, record);
                        ctx.recordResult(record);
                        loadResults(owner, false).catch(error => notify("成片已在当前页面，跨次打开的记录暂无法同步：" + error.message));
                    }
                    const failure = (data.status?.messages || []).filter(item => item[0] === "execution_error").at(-1)?.[1];
                    const reason = failure ? `<div class="o-warn"><strong>失败节点：${esc(failure.node_type || failure.node_id || "未知")}</strong><p>${esc(failure.exception_message || "请展开任务详情查看错误")}</p></div>` : "";
                    const media = videos.map(item => {
                        const url = viewURL(item);
                        return url ? `<video controls preload="metadata" style="max-width:100%;max-height:55vh" src="${esc(url)}"></video>` : "";
                    }).join("");
                    lastJobDialog = [promptId, data.state === "success" ? "导演台任务结果" : "导演台生成失败",
                        `<p>所属项目：${esc(owner || "旧记录未标注")} · 镜头：${esc(shotId)} · ${esc(recipe || "director")}</p>${reason}${media || "<p>本次没有可播放的视频条目。</p>"}<details><summary>任务详情与原始错误</summary><pre>${esc(JSON.stringify({ prompt_id: promptId, state: data.state, status: data.status, outputs: data.outputs }, null, 2))}</pre></details>`];
                    $("[data-service=\"job-status\"]").hidden = false;
                    if (data.state === "success" && videos.length && !belongsHere) {
                        showJobDialog(promptId, "原项目任务已完成", `<p>所属项目：${esc(owner || "旧记录未标注") }；没有写入当前镜头。请打开原项目查看成片。</p>${media}`);
                        notify("原项目任务已完成；当前项目的镜头结果未改动。");
                    } else if (data.state === "success" && videos.length) {
                        if ($("[data-dialog]").open && $("[data-dialog]").dataset.jobId === promptId) $("[data-dialog]").close();
                        notify("第 " + (ctx.doc().shots.findIndex(shot => shot.id === shotId) + 1) + " 镜已生成；点镜头卡或“镜头成片”即可直接播放。");
                    } else {
                        showJobDialog(promptId, data.state === "success" ? "生成完成但未找到视频" : "导演台生成失败",
                            `<p>${data.state === "success" ? "Core 完成，但输出没有可播放视频；任务保留。" : "Core 返回失败，原任务记录已保留。"} · ${esc(recipe || "director")}</p>${reason}${media || "<p class=\"o-muted\">本次没有可播放的视频条目。</p>"}<details><summary>任务详情与原始错误</summary><pre>${esc(JSON.stringify({ prompt_id: promptId, state: data.state, status: data.status, outputs: data.outputs }, null, 2))}</pre></details>`);
                        notify(data.state === "success" ? "未找到视频输出；可从任务状态查看详情，当前编辑保留。" : "生成失败；没有自动重采样，可从任务状态查看详情，当前编辑保留。");
                    }
                }
            } catch (error) {
                // A transient browser/Core disconnect must not resubmit the job.
                if (activeJobId !== promptId) return;
                if (error.status === 404) {
                    finish("unknown");
                    activeJobId = null;
                    forgetJob(promptId);
                    notify("Core 已找不到该任务记录；不会重复提交，原草稿仍保留。 ");
                    return;
                }
                if(lastPollError!==error.message)notify("生成状态暂时无法读取，任务不会重复提交：" + error.message);
                lastPollError=error.message;
            }
        };
        timer = setInterval(poll, 1500);
        await poll();
        return completion;
    }
    async function generate(variation = false) {
        if (busy) { showActiveJob(); return; }
        const field = $("[data-shot-seed]");
        if (!variation && field?.checkValidity && !field.checkValidity()) { field.reportValidity(); return; }
        busy = true;
        try {
            const shotId = ctx.current();
            const shot = ctx.doc().shots.find(item => item.id === shotId);
            let seed = directorSeed(shot);
            // Persist before submission. A lost response must retry this same seed and request.
            if (variation || shot.seed == null) {
                ctx.checkpoint();
                if (variation) seed = newDirectorSeed(seed);
                shot.seed = seed;
                if (field) { field.value = String(seed); field.setCustomValidity?.(""); }
                draft();
            }
            const project = envelope();
            const data = await submitGenerate(project, shotId, seed);
            rememberJob(data.prompt_id, data.recipe, shotId, project.id);
            if (project.id === projectId && ctx.doc().shots.some(shot => shot.id === shotId)) ctx.recordResult({ project_id: project.id, shot_id: shotId, prompt_id: data.prompt_id, state: "pending", outputs: {}, recipe: data.recipe });
            notify("已提交正式 Core 任务，断线只轮询原任务，不会重复提交。");
            await watchJob(data.prompt_id, data.recipe, shotId, project.id);
        } finally { busy = false; }
    }
    async function loadModels() { return request("models"); }
    async function prepareBundle(includeResults=true) {
        const project=envelope(),context=contextToken(),snapshot=JSON.stringify(project);
        const result=await request('bundles/prepare',{project,include_results:includeResults});
        if(context!==contextToken()||snapshot!==JSON.stringify(envelope()))return null;
        return result;
    }
    async function buildBundle(plan) {
        const path=`bundles/${encodeURIComponent(plan.project_id)}/${encodeURIComponent(plan.id)}`;
        const result=await request(path+'/build',{confirm_contents:true});
        return {...result,download_url:new URL(path+'/file',base).href};
    }
    async function uploadBundle(file,signal) {
        if(!file||!file.name.toLowerCase().endsWith('.zip'))throw Error('请选择导演台工程 ZIP');
        if(file.size>20*1024**3+12*1024**2)throw Error('工程包超过20GiB上限');
        const body=new FormData();body.append('file',file);
        const response=await fetch(new URL('bundle-imports',base),{method:'POST',credentials:'same-origin',body,signal});
        const result=await response.json();if(!response.ok)throw Error(result.error||'工程包检查失败');
        return {...result,new_project_id:id()};
    }
    async function applyBundle(plan) {
        // Persist the exact target before sending: uncertain retries must not create duplicates.
        const target={id:plan.id,new_project_id:plan.new_project_id};
        try{localStorage.setItem('t8director.lastBundleImport',JSON.stringify(target));}catch{notify('浏览器无法记住导入请求，请保留新项目 ID：'+target.new_project_id);}
        return request(`bundle-imports/${encodeURIComponent(plan.id)}/apply`,{new_project_id:plan.new_project_id,confirm_new_project:true});
    }
    async function lastBundleImport() {
        let target;try{target=JSON.parse(localStorage.getItem('t8director.lastBundleImport')||'null');}catch{return null;}
        if(!target?.id||!target?.new_project_id)return null;
        return {...await request('bundle-imports/'+encodeURIComponent(target.id)),new_project_id:target.new_project_id};
    }
    function filmUrls(result) {
        if(result.ready){
            const path=`films/${encodeURIComponent(result.project_id)}/${encodeURIComponent(result.id)}`;
            result.manifest_url=new URL(path,base).href;
            result.entries=result.entries.map((entry,index)=>({...entry,media_url:new URL(path+'/media/'+index,base).href}));
        }
        return result;
    }
    async function prepareFilm() {
        const project=envelope(),context=contextToken(),snapshot=JSON.stringify(project);
        const result=await request('films/prepare',{project});
        if(context!==contextToken()||snapshot!==JSON.stringify(envelope()))return null;
        return filmUrls(result);
    }
    async function prepareComparison(keys) {
        const original=envelope(),context=contextToken(),snapshot=JSON.stringify(original),shotId=ctx.current();
        const records=ctx.versions?.()||[];
        if(!Array.isArray(keys)||keys.length!==2||keys[0]===keys[1]||keys.some(key=>!records.some(row=>row.shot_id===shotId&&directorResultKey(row)===key&&row.state==='success'&&directorOutputVideos(row).length===1)))throw Error('请选择本镜两个不同的、各有一个成片的成功版本');
        const films=await Promise.all(keys.map(async key=>{
            const project=structuredClone(original),shot=project.doc.shots.find(row=>row.id===shotId);
            shot.adoptedResultId=key;delete shot.filmTrim;project.doc.shots=[shot];project.current=shotId;
            return filmUrls(await request('films/prepare',{project}));
        }));
        if(context!==contextToken()||shotId!==ctx.current()||snapshot!==JSON.stringify(envelope()))return null;
        const failed=films.find(film=>!film.ready);
        if(failed)throw Error((failed.errors||[]).map(error=>error.message).join('；')||'无法固定对比媒体');
        return films.map(film=>film.entries[0]);
    }
    async function prepareFrame(key) {
        const project=envelope(),context=contextToken(),snapshot=JSON.stringify(project),shotId=ctx.current();
        if(!(ctx.versions?.()||[]).some(row=>row.shot_id===shotId&&directorResultKey(row)===key&&row.state==='success'&&directorOutputVideos(row).length===1))throw Error('请选择本镜包含唯一成片的成功版本');
        const shot=project.doc.shots.find(row=>row.id===shotId);
        shot.adoptedResultId=key;delete shot.filmTrim;project.doc.shots=[shot];project.current=shotId;
        const result=filmUrls(await request('films/prepare',{project}));
        if(context!==contextToken()||shotId!==ctx.current()||snapshot!==JSON.stringify(envelope()))return null;
        if(!result.ready)throw Error((result.errors||[]).map(row=>row.message).join('；')||'无法准备成片');
        return result;
    }
    async function extractFrame(film,frame) {
        if(film.project_id!==projectId)throw Error('成片不属于当前项目');
        const result=await request(`films/${encodeURIComponent(projectId)}/${encodeURIComponent(film.id)}/frames`,{index:0,frame});
        return {...result,asset:{...result.asset,url:new URL('assets/'+result.asset.id,base).href}};
    }
    async function startFilmExport(film,options) {
        if(film.project_id!==projectId)throw Error('整片属于之前的项目，请回到对应项目再导出');
        const target={project_id:projectId,film_id:film.id,job_id:id()};
        try{localStorage.setItem('t8director.filmExport:'+projectId,JSON.stringify(target));}catch{notify('浏览器无法记住导出任务，请保留任务 ID：'+target.job_id);}
        await request(`films/${encodeURIComponent(target.project_id)}/${encodeURIComponent(target.film_id)}/exports/${encodeURIComponent(target.job_id)}`,options);
        return target;
    }
    async function filmExportStatus(target=null) {
        if(!target){try{target=JSON.parse(localStorage.getItem('t8director.filmExport:'+projectId)||'null');}catch{return null;}}
        if(!target||target.project_id!==projectId)return null;
        const origin=contextToken(),path=`films/${encodeURIComponent(target.project_id)}/${encodeURIComponent(target.film_id)}/exports/${encodeURIComponent(target.job_id)}`;
        const job=await request(path);
        if(origin!==contextToken())return null;
        return {...job,target,download_url:new URL(path+'/file',base).href};
    }
    async function showVersion(record) {
        const sequence = ++versionRequest;
        versionCopy=null;
        if (!record?.snapshot_available || !record.request_id) {
            showDialog("版本详情", "<p>此版本未保存完整配置快照，不能精确还原。成片仍可播放、下载与采用。</p>");
            return;
        }
        const context = contextToken(), current = ctx.current();
        const result = await request(`snapshots/${encodeURIComponent(projectId)}/${encodeURIComponent(record.request_id)}`);
        if (context !== contextToken() || current !== ctx.current() || sequence !== versionRequest) return;
        if ($('[data-dialog]').open||otherModalOpen()) { notify('版本配置已读取；请关闭当前编辑窗口后再次查看。'); return; }
        showDialog("版本配置快照", result.complete
            ? `<p>生成时保存的只读配置，不是当前草稿。种子：${esc(result.snapshot.seed)}；配方：${esc(result.snapshot.recipe)}。</p><button data-service="copy-version">复制此版设置为新草稿</button><p>新建独立工程，不覆盖当前页；只还原此镜实际种子，其它镜头保留当时草稿。模型文件或环境变化可能使结果不同。</p><details><summary>完整项目、素材身份与实际编译图</summary><pre>${esc(JSON.stringify(result.snapshot, null, 2))}</pre></details>`
            : `<p>${esc(result.reason)}</p>`);
        if(result.complete)versionCopy={owner:projectId,request:record.request_id,target:id(),context,sequence};
    }
    async function copyVersion(button) {
        const copy=versionCopy,dialog=$('[data-dialog]');
        if(!copy||copy.context!==contextToken())throw Error('版本窗口已过时，请重新打开版本详情');
        button.disabled=true;
        const html=dialog.innerHTML;
        try{
            const saved=await request(`snapshots/${encodeURIComponent(copy.owner)}/${encodeURIComponent(copy.request)}/copy`,{new_project_id:copy.target});
            if(copy!==versionCopy||copy.context!==contextToken()||!dialog.open||dialog.innerHTML!==html||otherModalOpen()){notify('版本新草稿已保存，可在“打开项目”中查找；当前草稿未改变。');return;}
            showDialog('版本新草稿已保存',`<p>当前工程未改变；新草稿已独立保存。不包含旧工程的成片采用状态，也没有提交生成。</p><button data-open-project="${esc(saved.id)}">打开版本新草稿</button>`);
            versionCopy=null;
        }finally{button.disabled=false;}
    }
    async function submitGenerate(project, shotId, seed) {
        const input = {project, shot_id:shotId, seed};
        const encoded = JSON.stringify(input);
        const fingerprintBytes = crypto.subtle ? await crypto.subtle.digest('SHA-256', new TextEncoder().encode(encoded)) : null;
        const fingerprint = fingerprintBytes ? [...new Uint8Array(fingerprintBytes)].map(value=>value.toString(16).padStart(2,'0')).join('') : encoded;
        const key = `t8director.request.${project.id}.${shotId}`;
        let pending = null;
        const raw = storageRead('sessionStorage',key);
        try { pending = pendingRequests.get(key) || (raw && JSON.parse(raw)); }
        catch { throw Error('原生成请求恢复记录损坏；请从任务列表核对，未创建新请求。'); }
        if(pending && !pending.request_id)throw Error('原生成请求身份损坏；未创建新请求，请保留本地记录并核对任务。');
        if(pending && !pending.input && pending.fingerprint!==fingerprint){
            // Upgrade old fingerprint-only recovery records from the immutable
            // server snapshot, never reconstruct the old input from a new draft.
            try{
                const original=await request(`snapshots/${encodeURIComponent(project.id)}/${encodeURIComponent(pending.request_id)}`);
                const source=original.snapshot;
                if(!original.complete||source?.project?.id!==project.id||source.shot_id!==shotId)throw Error('未取得完整原请求快照');
                pending.input={project:source.project,shot_id:source.shot_id,seed:source.seed};
            }catch(error){
                pendingRequestReview={key,request_id:pending.request_id,context:contextToken()};
                if(!otherModalOpen())showDialog('旧生成请求需核对',`<p>原请求 ID：${esc(pending.request_id)}</p><p>尚未取得原始配置：${esc(error.message)}。没有提交新任务。请先查看任务列表；仅在确认需要另起任务后放弃本页重试身份。</p><button data-service="task-list">查看任务列表</button><button data-service="abandon-request">已核对，放弃原重试身份</button>`);
                throw Error('旧生成请求尚未确认，未自动改用新配置；请核对原任务或在确认窗口显式放弃重试身份。');
            }
        }
        if (pending?.input && (pending.input.project?.id !== project.id || pending.input.shot_id !== shotId)) throw Error('原请求身份与当前镜头不符，未提交任务。');
        if (!pending) pending = {request_id:id(),fingerprint,input:structuredClone(input)};
        else if (!pending.input) pending.input = structuredClone(input);
        if (stableJSON(pending.input) !== stableJSON(input)) notify('正在核对上次失联请求：使用原配置和原请求 ID；当前新编辑仍保留，不会另起生成。');
        pendingRequests.set(key,pending);
        const cached = storageWrite('sessionStorage',key,JSON.stringify(pending));
        if (!cached) notify('无法持久保护生成请求；本页内仍可安全重试，请勿刷新，任务 ID 将在响应后显示。');
        try {
            const data = await request('generate', {...pending.input,client_id:tab,request_id:pending.request_id});
            pendingRequests.delete(key); storageRemove('sessionStorage',key);
            return data;
        } catch (error) {
            // Only an explicit server confirmation made before reservation/queue
            // may release an ID. Network errors and ambiguous 4xx/5xx retain it.
            if(error.data?.submission_state==='not_submitted') {pendingRequests.delete(key);storageRemove('sessionStorage',key);}
            if(error.data?.prompt_id)error.message += ' · 原任务 ID：' + error.data.prompt_id;
            throw error;
        }
    }
    function showBatchStatus(status) {
        const rows = (status.items || []).map((item, index) =>
            `<li>批次 ${index + 1}：${esc(directorTaskLabel(item.state))}${ctx.doc().shots.some(shot=>shot.id===item.shot_id)?` <button data-service="locate-task" data-task-shot="${esc(item.shot_id)}">定位镜头</button>`:''}${item.prompt_id ? " · 任务 " + esc(item.prompt_id) : ""}</li>`).join("");
        const note = status.complete ? "全部镜头的输出文件身份已核对；画面与声音仍需观看验收。" :
            "只会在明确点击继续后处理冻结批次；未知／损坏状态不会自动重发。";
        const uncertain = (status.items || []).some(item => ["queued", "running", "unknown"].includes(item.state));
        const first = status.items?.[status.next_index];
        showDialog("全部生成批次", `<p>批次 ID：${esc(status.id)} · ${esc(note)}</p><ol>${rows}</ol>` +
            (status.complete ? "" : `<button data-service="resume-batch">继续剩余镜头</button>` +
                (first?.retry_available ? `<button data-service="retry-batch">确认弃用旧尝试并重试第 ${status.next_index + 1} 镜</button>` : "") +
                (uncertain ? "" : `<button data-service="new-batch">另起全新批次（重新生成全部）</button>`)));
    }
    async function taskSnapshot(signal) {
        const origin=contextToken(),owner=projectId,batchId=savedBatchId(owner);
        const query=new URLSearchParams(ctx.doc().shots.map(shot=>['shot',shot.id])).toString();
        const data=await request('results/'+encodeURIComponent(owner)+(query?'?'+query:''),undefined,'GET',signal);
        let batch=null,batchError='';
        if(batchId)try{batch=await request('batches/'+encodeURIComponent(batchId),undefined,'GET',signal);}catch(error){if(error.name==='AbortError')throw error;batchError=error.message;}
        if(origin!==contextToken())return null;
        return {records:data.results||[],batch,batchError};
    }
    async function showTasks() {
        if(ctx.openTasks)return ctx.openTasks();
        const sequence=++tasksRequest,origin=contextToken(),owner=projectId,previousModal=$('[data-dialog]').innerHTML;
        const records=await loadResults(owner);
        const batchId=savedBatchId(owner);
        let batch=null,batchError='';
        if(batchId)try{batch=await request('batches/'+encodeURIComponent(batchId));}catch(error){batchError=error.message;}
        if(origin!==contextToken()||sequence!==tasksRequest)return;
        if(otherModalOpen()||($('[data-dialog]').open&&$('[data-dialog]').innerHTML!==previousModal)){notify('任务状态已更新；请关闭当前编辑窗口后再打开任务列表。');return;}
        const shots=ctx.doc().shots;
        const locate=(id)=>{const i=shots.findIndex(shot=>shot.id===id);return i<0?'原镜头已不在当前草稿':`<button data-service="locate-task" data-task-shot="${esc(id)}">第 ${i+1} 镜 · ${esc(shots[i].name||'未命名')}</button>`;};
        const rows=records.map(record=>`<li>${locate(record.shot_id)} <span>${esc(directorTaskLabel(record.state))}</span><small>${record.submitted_at?esc(new Date(record.submitted_at*1000).toLocaleString()):'时间未记录'}</small></li>`).join('');
        const frozen=batch?`<details open><summary>冻结批次 · ${batch.items.filter(item=>item.state==='success').length}/${batch.items.length}</summary><p class="o-tip">以提交时的顺序与种子继续，不跟随当前草稿重排。</p><ol>${batch.items.map((item,index)=>`<li>批次 ${index+1} · ${locate(item.shot_id)} · ${esc(directorTaskLabel(item.state))}</li>`).join('')}</ol>${batch.complete?'':`<button data-service="resume-batch">核对并继续剩余镜头</button>`}</details>`:'';
        showDialog('任务列表',`<p>仅当前项目。定位不会中断任务；暂停批次请使用底栏“本镜完成后暂停”。</p>${frozen}${batchError?`<p>旧批次暂不可读：${esc(batchError)}；没有自动重发。</p>`:''}<ul class="w-task-list">${rows||'<li>尚无生成记录。</li>'}</ul><button data-service="task-list">刷新任务状态</button>`);
    }
    async function refreshBatchButton(owner = projectId) {
        const button = $(".o-footer [data-service=\"resume-batch\"]");
        if (!button) return;
        const batchId = savedBatchId(owner);
        button.hidden = !batchId || owner !== projectId;
        if (!batchId || owner !== projectId) return;
        try {
            const state = await request("batches/" + encodeURIComponent(batchId));
            if (owner !== projectId || savedBatchId(owner) !== batchId) return;
            button.hidden = !!state.complete;
            button.textContent = state.complete ? "全部已完成" : `继续剩余镜头 · ${state.items.filter(item => item.state === "success").length}/${state.items.length}`;
        } catch (error) {
            if (owner === projectId && savedBatchId(owner) === batchId) { button.hidden = false; button.textContent = "核对旧批次"; notify("批次状态暂不可读；没有自动提交任务：" + error.message); }
        }
    }
    async function runBatch(batchId) {
        if (busy) { showActiveJob(); return; }
        const origin = contextToken();
        busy = true;
        activeBatch = batchId; pauseBatch = false;
        const pauseButton = $('[data-service="pause-batch"]');
        pauseButton.hidden = false; pauseButton.disabled = false;
        pauseButton.textContent = '本镜完成后暂停';
        try {
            // A single click is the authorization for this sequential run.
            // Reloading the page only reads status and never enters this loop.
            for (;;) {
                const status = await request("batches/" + encodeURIComponent(batchId));
                if (origin !== contextToken() || status.project_id !== projectId || savedBatchId(status.project_id) !== batchId) {
                    notify("批次所属项目已切换；原任务不会写入当前项目，也不会继续提交后续镜头。"); return;
                }
                if (status.complete) { notify(`全部 ${status.items.length} 镜有输出文件，身份已核对；请实际观看音画。`); return; }
                if (pauseBatch) { notify('批次已暂停；已完成镜头与冻结配置保留，可继续剩余镜头。'); return; }
                const index = status.next_index, row = status.items[index];
                if (row.state !== "not_submitted" && !["queued", "running"].includes(row.state)) {
                    showBatchStatus(status);
                    notify(`第 ${index + 1} 镜为${directorTaskLabel(row.state)}，需要先核对，未提交后续镜头。`); return;
                }
                let promptId = row.prompt_id, recipe = "director";
                if (row.state === "not_submitted") {
                    // The server verifies all earlier media under the submission lock.
                    // The immutable server snapshot and deterministic request ID win
                    // over any edits made to the current browser draft.
                    const data = await request("batches/" + encodeURIComponent(batchId) + "/continue", { client_id: tab });
                    promptId = data.prompt_id; recipe = data.recipe;
                    if (!promptId) { const updated=await request("batches/" + encodeURIComponent(batchId)); if(origin===contextToken())showBatchStatus(updated); return; }
                    // Already-submitted work must remain tracked even after a switch;
                    // only the subsequent submission intent is invalidated.
                    if (origin === contextToken() && status.project_id === projectId && ctx.doc().shots.some(shot => shot.id === row.shot_id)) {
                        ctx.recordResult({ project_id: status.project_id, shot_id: row.shot_id, prompt_id: promptId, state: "pending", outputs: {}, recipe });
                    }
                }
                rememberJob(promptId, recipe, row.shot_id, status.project_id);
                const observed = await watchJob(promptId, recipe, row.shot_id, status.project_id);
                if (origin !== contextToken()) return;
                if (observed !== "success") {
                    notify(`第 ${index + 1} 镜未完成；后续镜头未提交。`); return;
                }
                // Never trust the browser's success alone: next iteration checks
                // Core receipt, media bytes, and previous SHA on the server.
            }
        } catch (error) {
            notify("批次执行暂停，未自动重试提交：" + error.message);
        } finally { busy = false; activeBatch = null; pauseButton.hidden = true; await refreshBatchButton(); }
    }
    async function generateAll({fresh = false, scope = 'all'} = {}) {
        if (busy) { showActiveJob(); return; }
        busy = true;
        const batch = envelope();
        const selectedIds = new Set(ctx.selectedShotIds?.() || []);
        const frozenVersions = structuredClone(ctx.versions?.() || []);
        let shots = [...batch.doc.shots];
        let batchId = null, comparisonNote = '';
        const origin = contextToken();
        try {
            const oldId = savedBatchId(batch.id);
            if (oldId && !fresh) {
                let old;
                try { old = await request("batches/" + encodeURIComponent(oldId)); }
                catch (error) {
                    if (error.status !== 404) throw error;
                    notify("服务端确认旧批次不存在；将重新建立当前项目的批次。");
                }
                if (origin !== contextToken()) return;
                if (old && !old.complete) { showBatchStatus(old); notify("仍有未完成的冻结批次；请先核对或明确继续。新草稿不会暗中替换旧批次。"); return; }
            }
            if(scope==='selected')shots=shots.filter(shot=>selectedIds.has(shot.id));
            if(scope==='ungenerated')shots=shots.filter(shot=>!frozenVersions.some(item=>item.shot_id===shot.id&&item.state==='success'&&directorOutputVideos(item).length));
            if(scope==='modified'){
                const changed=[];let unknown=0;
                for(const shot of shots){
                    const versions=frozenVersions.filter(item=>item.shot_id===shot.id);
                    const result=selectDirectorResult(versions,shot.adoptedResultId);
                    if(!result?.snapshot_available||!result.request_id){unknown++;continue;}
                    const source=await request(`snapshots/${encodeURIComponent(batch.id)}/${encodeURIComponent(result.request_id)}`);
                    const oldShot=source.snapshot?.project?.doc?.shots.find(item=>item.id===shot.id);
                    if(!source.complete||!oldShot){unknown++;continue;}
                    if(directorShotInputKey(batch.doc,shot,batch.assets)!==directorShotInputKey(source.snapshot.project.doc,{...oldShot,seed:source.snapshot.seed},source.snapshot.project.assets))changed.push(shot);
                }
                shots=changed;
                if(unknown)comparisonNote=`${unknown} 镜没有完整对照快照，未自动判定为已修改；可手动选择生成。`;
            }
            if(!shots.length){notify('当前范围没有需要生成的镜头；没有提交任务。'+comparisonNote);return;}
            if(origin!==contextToken()){notify('项目已切换，尚未提交批次。');return;}
            let features;
            try{features=await request('batch-features');}catch{notify('当前 Core 尚未确认新版批次协议。请保存项目，待现有任务结束后重启 Core；本次未提交生成。');return;}
            if(features.selection_version!==2){notify('Core 与页面批次版本不匹配；请更新并重启 Core。本次未提交生成。');return;}
            if(origin!==contextToken()){notify('项目已切换，尚未提交批次。');return;}
            const shot_ids=shots.map(shot=>shot.id);
            const seed_map=Object.fromEntries(shots.map(shot=>[shot.id,directorSeed(shot)]));
            const report = scope==='all' ? await request("compile", { project: batch }) : await request("compile", { project: batch, shot_ids });
            if(origin!==contextToken()){notify('项目已切换，尚未提交批次。');return;}
            if (!report.ready) {
                const issues = report.errors.map(error => {
                    const index = batch.doc.shots.findIndex(shot => shot.id === error.shot_id);
                    return (index >= 0 ? `第 ${index + 1} 镜「${batch.doc.shots[index].name}」：` : "全片素材：") + error.message;
                });
                showDialog("全部生成前检查 · 尚未提交任务", "<p>本次范围："+shots.length+" 镜。以下问题属于对应镜头；如果只想生成当前镜头，请关闭后使用“生成当前镜头”。</p><ul>" + issues.map(text => "<li>" + esc(text) + "</li>").join("") + "</ul>");
                notify("全片尚有未完成镜头，未提交任何生成任务；可以单独生成当前镜头。");
                return;
            }
            batchId = id();
            saveBatchId(batch.id, batchId);
            const seed = 26091901;
            await request("batches", { batch_id: batchId, project: batch, seed, shot_ids, seed_map });
            if(comparisonNote)notify(comparisonNote);
        } catch (error) {
            notify("批次创建未确认；已保留批次 ID，可核对后继续，未自动重发：" + error.message);
            return;
        } finally { busy = false; }
        if (batchId && origin === contextToken()) await runBatch(batchId);
        else if (batchId) notify('项目已切换；冻结批次链接保留，未自动提交镜头，请核对后明确继续。');
    }
    async function exportGraph(kind) {
        const data = await request("export", { project: envelope(), shot_id: ctx.current() });
        download(kind === "workflow" ? "director-d1-preflight.workflow.json" : "director-d1-preflight.api.json", kind === "workflow" ? data.workflow : data.api_snapshot);
        notify("已导出原生 D1 CPU预检图／API快照，不是GPU生成工作流；不会排队。");
    }
    async function showCapabilities() {
        const data = await request("capabilities");
        const rows = (data.capabilities || []).map(item => {
            const state = item.state === "ready" ? "已注册入口" : "缺少入口节点";
            return `<article class="o-asset"><h3>${esc(item.label)} · ${esc(state)}</h3><p class="o-muted">${esc(item.note)}</p><small>${esc(item.entry_nodes.join(" · "))}</small><p class="o-tip">${esc(item.execution)}</p></article>`;
        }).join("");
        showDialog("D3 配套能力检查", `<p>这里列出当前 Core 已注册的正式入口；“已注册入口”不等于 GPU 成片或人审通过。点击后仍需进入对应原生工作流，避免把不同状态合同强行拼成一个万能按钮。</p><div class="o-grid">${rows}</div>`);
    }
    async function showD3Preflight() {
        const p = envelope(), epoch = projectEpoch, sequence = ++d3Request, snapshot = JSON.stringify(p);
        const data = await request("d3/preflight", { project: p, shot_id: p.current });
        if (epoch !== projectEpoch || sequence !== d3Request || snapshot !== JSON.stringify(envelope())) return;
        const rows = (data.capabilities || []).map(item => {
            const ready = item.state === "ready_for_native_workflow";
            const deps = (item.dependencies || []).map(dep => `${dep.ok ? "✓" : "!"} ${dep.detail}`).join("；");
            const files = (item.workflows || []).map(file => `<div class="o-row o-between"><code>${esc(file)}</code><button data-d3-handoff="${esc(item.id)}" data-d3-file="${esc(file)}">${file.toLowerCase().endsWith(".json") ? "下载工作流" : "下载说明"}</button></div>`).join("");
            const contract = item.contract || {};
            const steps = (contract.steps || []).map((step, index) => (index + 1) + ". " + step).join("；");
            const requires = (contract.requires || []).join("、");
            return `<article class="o-asset"><h3>${esc(item.label)} · ${ready ? "可交接原生路线" : "需先补依赖"}</h3><p>${esc(deps)}</p><p class="o-tip">继续步骤：${esc(steps || item.next_action)}</p><p class="o-tip">需要：${esc(requires || "由原生入口继续检查")}</p><div class="o-grid">${files}</div><div class="o-row o-gap"><button data-d3-package="${esc(item.id)}">导出当前镜头路线包</button></div><p class="o-tip">${esc(contract.boundary || item.next_action)}</p></article>`;
        }).join("");
        showDialog("当前镜头 · D3 逐项预检", `<p>镜头 ${esc(p.current)} 已按服务端 project/media_map 编译。这里的“可交接”只表示入口、依赖和项目合同通过，不会偷偷排队，也不代表 GPU 或人审通过。</p><div class="o-grid">${rows}</div><pre>${esc(JSON.stringify({ schema: data.schema, compile: data.compile, warning: data.warning }, null, 2))}</pre>`);
    }
    async function compileD3() {
        const data = await request("d3/compile", { project: envelope(), shot_id: ctx.current(), seed: Number(storageRead('sessionStorage', 't8director.seed') || 26091901) });
        const types = Object.entries(data.nodes || {}).map(([id, node]) => id + ": " + node.class_type).join("\n");
        showDialog("D3 原生图已编译 · 未排队", "<p>路线：" + esc((data.d3_routes || []).join("、") || "默认原生") + "</p><p class=\"o-tip\">这里只证明节点图和依赖可以编译，不会加载模型、不提交队列，也不代表 GPU 或感知质量通过。</p><pre>" + esc(types + "\n\n" + JSON.stringify({ recipe: data.recipe, warning: data.warning }, null, 2)) + "</pre>");
    }
    async function exportEditableSplit() {
        const project = envelope(), origin = contextToken(), shot_id = ctx.current();
        const seed = Number(storageRead('sessionStorage', 't8director.seed') || 26091901);
        const data = await request("d3/editable-split-workflow", { project, shot_id, seed });
        if (origin !== contextToken()) { notify('项目已切换，未下载过时的分离图。'); return; }
        const suffix = data.recipe.includes('tail_resume') ? 'resume-tail' : 'full';
        download('director-hyperflow-separate-' + suffix + '.workflow.json', data.workflow);
        notify('已导出当前镜头的可编辑分离式图；未排队。仅 TAIL 恢复图仍需真实 HEAD 回执，画质与音画待验收。');
    }
    async function handoffD3(capability, file) {
        const data = await request("d3/handoff", { capability, file });
        downloadBytes(data.name, data.text ?? JSON.stringify(data.content, null, 2));
        notify("已下载原生路线副本：" + data.name + "；不会自动映射当前镜头或排队。");
    }
    async function packageD3(capability) {
        const data = await request("d3/package", { capability, project: envelope(), shot_id: ctx.current() });
        const stamp = new Date().toISOString().replace(/[:.]/g, "-");
        downloadBytes("director-" + capability + "-route-package-" + stamp + ".json", JSON.stringify(data, null, 2));
        notify("已导出 " + capability + " 路线包：包含当前项目快照、原生工作流和继续步骤；不会上传媒体或排队。");
    }
    async function openList() {
        const data = await request("projects");
        showDialog("打开服务端项目", `<p>先保存当前稿；载入会替换当前编辑页，但不删除任何素材或作品。</p><div class="o-grid">${data.projects.map(p => `<button data-open-project="${esc(p.id)}" ${p.error ? "disabled" : ""}>${esc(p.title || p.error)} · v${p.revision ?? "?"}</button>`).join("") || "还没有服务端项目，请先保存。"}</div>`);
    }
    function preserveUnknown(value, bytes, name) {
        importFile = { bytes, name };
        showDialog("未知工作流 · 原样保留／只读", `<p>这不是导演台 project.json，不会猜测或覆盖镜头。可下载原文件，再返回 ComfyUI 画布打开。</p><button data-service="original">下载原图（不改写）</button><pre>${esc(JSON.stringify(value, null, 2))}</pre>`);
    }
    $(".o-project .o-row").insertAdjacentHTML("afterbegin", '<button data-action="model-settings" aria-haspopup="dialog">模型设置</button><button data-service="save">保存项目</button><button data-service="copy">另存副本</button><button data-service="open">打开项目</button><button data-service="project">导出项目</button><button data-service="import">导入JSON</button>');
    $(".o-project h3").outerHTML = `<label>项目名称<input data-project-title aria-label="项目名称" value="${esc(title)}"></label>`;
    $(".o-footer .o-row").insertAdjacentHTML("beforeend", '<button data-service="capabilities">D3能力检查</button><button data-service="d3-preflight">当前镜头D3预检</button><button data-service="d3-compile">编译D3图</button><button data-service="split-workflow">导出 HyperFlow 分离图 EXP</button><button data-service="workflow">导出预检工作流</button><button data-service="api">导出API快照</button>');
    $(".o-footer .o-row").insertAdjacentHTML("beforeend", '<button data-service="job-status" hidden>查看进行中任务</button>');
    $(".o-footer .o-row").insertAdjacentHTML("beforeend", '<button data-service="resume-batch" hidden>继续剩余镜头</button>');
    $(".o-footer .o-row").insertAdjacentHTML("beforeend", '<button data-service="pause-batch" hidden>本镜完成后暂停</button>');
    $(".o-project .o-row").insertAdjacentHTML("beforeend", '<button data-service="reconnect" title="重新关联缺失的图片、视频或录音；不会删除文件或成品">重连素材</button>');
    const importer = document.createElement("input"); importer.type = "file"; importer.accept = ".json"; importer.hidden = true; ctx.root.append(importer);
    importer.onchange = async () => {
        const ticket = beginProjectLoad();
        try {
            const file = importer.files[0];
            if (!file) return;
            const bytes = new Uint8Array(await file.arrayBuffer());
            if (!acceptProjectLoad(ticket)) return;
            let p = JSON.parse(new TextDecoder().decode(bytes));
            if (p.schema !== "t8.minimax_h3.director_project") { preserveUnknown(p, bytes, file.name); return; }
            p = (await request("validate", { project: p })).project; // One authoritative lossless migration.
            if (!acceptProjectLoad(ticket)) return;
            await request("compile", { project: p }); // Validation, including versions; missing media may remain draft.
            if (!acceptProjectLoad(ticket)) return;
            switchProject(p, ticket, "项目已导入草稿，尚未覆盖服务端；保存时检查版本冲突。");
        } catch (e) { notify("导入失败，当前项目保留：" + e.message); }
    };
    ctx.root.addEventListener("input", e => { editEpoch++; if (e.target.matches("[data-project-title]")) { title = e.target.value; draft(); } else if (e.target.matches("[data-global],[data-field],[data-event]")) draft(); });
    ctx.root.addEventListener("change", e => {
        if (!e.target.matches("[data-shot-seed]")) return;
        try {
            const seed = directorSeed({seed: e.target.value});
            const shot = ctx.doc().shots.find(item => item.id === ctx.current());
            if (shot.seed !== seed) { ctx.checkpoint(); shot.seed = seed; editEpoch++; draft(); }
            e.target.setCustomValidity("");
        } catch (error) { e.target.setCustomValidity(error.message); e.target.reportValidity(); notify(error.message); }
    });
    ctx.root.addEventListener("click", async e => {
        const b = e.target.closest("button"); if (!b || b.disabled) return;
        const action = b.dataset.service;
        if (!["check", "generate", "generate-all", "prompts"].includes(b.dataset.action) && !action && !b.dataset.openProject && !b.dataset.reconnectAsset && !b.dataset.d3Handoff && !b.dataset.d3Package) return;
        e.preventDefault(); e.stopImmediatePropagation();
        try {
            if (b.dataset.action === "generate") await generate();
            else if (action === "new-variation") await generate(true);
            else if (action === "task-list") await showTasks();
            else if (action === "copy-version") await copyVersion(b);
            else if (action === "locate-task") {
                if(!ctx.doc().shots.some(shot=>shot.id===b.dataset.taskShot)){notify('该镜头不在当前草稿，冻结任务仍保留。');return;}
                $('[data-dialog]').close();ctx.locateIssue?.({shot_id:b.dataset.taskShot,field:'prompt'});
            }
            else if (action === "pause-batch") {
                if (!activeBatch) return;
                pauseBatch = true; b.disabled = true; b.textContent = '将在本镜完成后暂停';
                notify('当前镜头继续完成，之后不再提交下一镜；不会中断 Core 或其他任务。');
            }
            else if (action === "locate-issue") {
                const issue = latest?.errors?.[Number(b.dataset.issueIndex)];
                if (!issue) { notify("检查已过时，请重新检查当前镜头。"); return; }
                $("[data-dialog]").close();
                ctx.locateIssue?.(issue);
            }
            else if (b.dataset.action === "generate-all") await generateAll();
            else if (action === "generate-selected") await generateAll({scope:'selected'});
            else if (action === "generate-ungenerated") await generateAll({scope:'ungenerated'});
            else if (action === "generate-modified") await generateAll({scope:'modified'});
            else if (action === "resume-batch") {
                const batchId = savedBatchId(projectId);
                if (!batchId) { notify("当前项目没有待核对的批次。"); return; }
                if (busy) { showActiveJob(); return; }
                await runBatch(batchId);
            }
            else if (action === "new-batch") {
                const owner = projectId, context = contextToken();
                const previous = savedBatchId(owner);
                if (!previous) return;
                const state = await request("batches/" + encodeURIComponent(previous));
                if (context !== contextToken() || savedBatchId(owner) !== previous) return;
                if (state.items.some(item => ["queued", "running", "unknown"].includes(item.state))) {
                    notify("旧批次还有未确认的任务；不能另起可能重复生成的新批次。"); return;
                }
                if (!confirm("新批次会按当前草稿从第一镜重新生成；原批次及其成片仍保留。确定继续？")) return;
                if (context !== contextToken() || savedBatchId(owner) !== previous) return;
                await generateAll({fresh: true});
            }
            else if (action === "retry-batch") {
                const owner = projectId, context = contextToken();
                const batchId = savedBatchId(owner);
                if (!batchId) return;
                const state = await request("batches/" + encodeURIComponent(batchId));
                if (context !== contextToken() || savedBatchId(owner) !== batchId) return;
                const row = state.items?.[state.next_index];
                if (!row?.retry_available) { notify("旧尝试仍未确定可弃用；没有重复提交。"); return; }
                if (!confirm("仅重试本镜，已完成镜头不会重算。旧回执保留；若另一 Core 仍在生成，可能产生重复作品。确定已停止旧任务并弃用这次尝试？")) return;
                if (context !== contextToken() || savedBatchId(owner) !== batchId) return;
                await request("batches/" + encodeURIComponent(batchId) + "/retry", {confirm_abandoned:true});
                if (context === contextToken() && savedBatchId(owner) === batchId) await runBatch(batchId);
            }
            else if (["check", "prompts"].includes(b.dataset.action)) await compile();
            else if (b.dataset.openProject) {
                if (!confirm("载入将替换本页草稿。未保存内容可先导出项目，是否继续？")) return;
                const sourceDialog=b.closest?.('dialog');
                const ticket = beginProjectLoad();
                const p = await request("projects/" + b.dataset.openProject);
                if (!acceptProjectLoad(ticket)) return;
                switchProject(p, ticket);
                if (!pendingProjectSwitch) sourceDialog?.close();
            } else if (action === 'backup-export' || action === 'backup-continue') {
                const pending = pendingProjectSwitch;
                if (!pending || !acceptProjectLoad(pending.ticket)) { pendingProjectSwitch=null; notify('当前草稿或载入目标已变化，请重新打开或导入项目。'); return; }
                if (action === 'backup-export') {
                    downloadBytes('director-unsaved-backup.project.json', pending.ticket.snapshot);
                    pending.exported = true; showBackupRecovery();
                } else if (pending.exported && confirm('请确认备份文件已经保存到电脑。继续将替换本页草稿，浏览器本地备份仍不可用。是否继续？')) {
                    if (acceptProjectLoad(pending.ticket)) finishProjectSwitch(pending.p, pending.note);
                }
            } else if (b.dataset.reconnectAsset) { reconnectId = b.dataset.reconnectAsset; reconnectTarget = { id: reconnectId, context: contextToken() }; reconnectPicker.value = ""; reconnectPicker.click(); }
            else if(action === 'abandon-request'){
                const review=pendingRequestReview;
                if(!review||review.context!==contextToken())throw Error('项目已切换，请重新核对原请求。');
                if(!confirm('原任务可能已经在 Core 运行。放弃后再次生成可能产生重复任务和额外耗时；本操作不会取消原任务。确认已核对并放弃本页重试身份？'))return;
                pendingRequests.delete(review.key);storageRemove('sessionStorage',review.key);pendingRequestReview=null;
                $('[data-dialog]').close();notify('已显式放弃原重试身份，没有提交或取消任务；需要新任务时请再次点击生成。');
            }
            else if (action === "save" || action === "copy") await save(action === "copy");
            else if (action === "job-status") showActiveJob();
            else if (action === "cancel-upload") {
                if (activeUpload) activeUpload.abort();
                else uploadNotice("当前没有正在上传的素材。", false);
            }
            else if (action === "cancel-job") {
                if (!activeJobId) { notify("当前没有可取消的导演台任务。"); return; }
                const id = activeJobId;
                const watcher = stopWatchingJob;
                let result;
                try { result = await request("jobs/" + encodeURIComponent(id) + "/cancel", {}); }
                catch (error) {
                    if (activeJobId === id && stopWatchingJob === watcher) notify('取消请求未确认；保留任务身份并继续跟踪，可核对后重试：' + error.message);
                    return;
                }
                if (activeJobId !== id || stopWatchingJob !== watcher) return;
                if (result.deleted_from_queue || result.interrupted) {
                    stopWatchingJob?.();
                    activeJobId = null;
                    forgetJob(id);
                    notify("已请求取消当前任务；不会影响其他队列任务。");
                } else notify("未确认取消成功；保留任务身份并继续核对状态。");
            }
            else if (action === "open") await openList();
            else if (action === "capabilities") await showCapabilities();
            else if (action === "d3-preflight") await showD3Preflight();
            else if (action === "d3-compile") await compileD3();
            else if (b.dataset.d3Handoff) await handoffD3(b.dataset.d3Handoff, b.dataset.d3File);
            else if (b.dataset.d3Package) await packageD3(b.dataset.d3Package);
            else if (action === "project") download("director.project.json", envelope());
            else if (action === "workflow" || action === "api") await exportGraph(action);
            else if (action === "split-workflow") await exportEditableSplit();
            else if (action === "import") { importer.value = ""; importer.click(); }
            else if (action === "original") downloadBytes(importFile.name, importFile.bytes);
            else if (action === "reconnect") showDialog("显式重连素材", `<p>请选择原素材对应的卡片再上传替代文件，全部镜头引用会同步到新身份；不覆盖旧文件，可撤销。</p><div class="o-grid">${[...ctx.assets().values()].map(a => `<button data-reconnect-asset="${a.id}">${esc(a.name)} · ${a.id.slice(0,8)}</button>`).join("")}</div>`);
        } catch (error) { notify(error.message); if (error.data?.report) showDialog("导出未完成 · 可以修正后重试", `<pre>${esc(JSON.stringify(error.data.report.errors, null, 2))}</pre>`); }
    }, true);
    const reconnectPicker = document.createElement("input"); reconnectPicker.type = "file"; reconnectPicker.accept = "image/*,video/*,audio/*"; reconnectPicker.hidden = true; ctx.root.append(reconnectPicker);
    reconnectPicker.onchange = async () => {
        const target = reconnectTarget;
        try {
            if (!target || target.context !== contextToken()) { notify("项目已切换，未上传或重连素材。"); return; }
            const old = ctx.assets().get(target.id);
            if (!old) throw Error("原素材已不在当前项目，请重新选择。");
            const next = await upload(reconnectPicker.files[0]);
            if (target.context !== contextToken() || target !== reconnectTarget) { notify("项目或重连目标已切换；已上传素材未绑定，当前项目保持不变。"); return; }
            if (old.kind !== next.kind) throw Error("重连须同一种媒体；本次新上传仍在服务端，不覆盖原资产");
            ctx.checkpoint(); const doc = ctx.doc();
            ctx.assets().set(next.id, next);
            doc.sharedRefs = doc.sharedRefs.map(a => a === target.id ? next.id : a);
            for (const s of doc.shots) {
                for (const key of ["first", "last", "audio", "selected"]) if (s[key] === target.id) s[key] = next.id;
                for (const key of ["refs", "tray"]) s[key] = (s[key] || []).map(a => a === target.id ? next.id : a);
                if (s.audio === next.id) { s.start = 0; s.end = next.duration; }
                s.rev++;
            }
            ctx.remapCreationAsset?.(target.id,next.id);
            // Keep the old library identity too: undo restores its references, not filesystem bytes.
            ctx.render(); draft(); $("[data-dialog]").close(); notify("重连完成，可撤销；原服务端文件没有被删或覆盖。");
        } catch (error) { notify(error.message); }
    };
    const restore = async () => {
        const pending = pendingJob();
        // Lock before any asynchronous project loading, not only after polling starts.
        if (pending) busy = true;
        try {
            await scopeReady;
            const specific = new URL(location.href).searchParams.get("project_id");
            const local = scopeFallbackDraft || storageRead('localStorage',draftKey);
            scopeFallbackDraft = null;
            const last = storageRead('localStorage','t8director.lastProject');
            if (specific) {
                let localProject = null;
                try { localProject = local && JSON.parse(local); } catch { /* Preserve corrupt bytes below. */ }
                if (localProject?.id === specific) {
                    hydrate(localProject);
                    $("[data-save]").textContent = `本地恢复副本 · 基于版本 ${revision} · 未核对服务端`;
                    notify("已恢复本项目的本地副本（可能包含未保存编辑）；尚未核对服务端最新内容，保存时会检查版本冲突。");
                } else {
                    // A different URL must not silently destroy the previous tab draft.
                    if (local) localStorage.setItem(draftKey + ":backup:" + (localProject?.id || "unreadable"), local);
                    hydrate(await request("projects/" + specific));
                    storageWrite('localStorage', draftKey, JSON.stringify(envelope()));
                }
            }
            else if (local) {
                hydrate(JSON.parse(local));
                $("[data-save]").textContent = `本地恢复副本 · 基于版本 ${revision} · 未核对服务端`;
                notify("已恢复本标签的本地副本（可能包含未保存编辑）；尚未核对服务端最新内容，保存时会检查版本冲突。");
            }
            else if (last) { hydrate(await request("projects/" + last)); notify("已从服务端恢复项目与素材，不需要重传。"); }
            else draft();
        } catch (error) { notify("自动恢复未完成，请用打开项目或导入备份：" + error.message); }
        finally {ctx.root.inert=false;}
        parent.postMessage({ type: "t8-director:ready" }, location.origin);
        refreshBatchButton().catch(error => notify("批次读取失败；没有自动提交：" + error.message));
        if (pending) {
            // Resume only status polling.  A refresh must never call /generate again.
            setTimeout(() => watchJob(pending.prompt_id, pending.recipe, pending.shot_id, pending.project_id).catch(error => notify("任务恢复失败，但不会重复提交：" + error.message)).finally(() => { busy = false; refreshBatchButton().catch(() => {}); }), 0);
        }
    };
    window.addEventListener("message", event => {
        if (event.origin !== location.origin || event.source !== parent || event.data?.type !== "t8-director:init") return;
        try { hydrate(event.data.project); draft(); notify("已恢复当前节点项目快照，保存前会核对服务端版本。"); }
        catch (error) { notify("节点快照未载入，原JSON保留：" + error.message); }
    });
    const checkedCanvas = () => latest && latestSnapshot === JSON.stringify(envelope()) ? latest.shots?.find(shot=>shot.id===ctx.current())?.canvas : null;
    return { draft, upload, restore, envelope, editorRecovery, compile, loadModels, showVersion, taskSnapshot, prepareFilm, prepareComparison, prepareFrame, extractFrame, prepareBundle, buildBundle, uploadBundle, applyBundle, lastBundleImport, startFilmExport, filmExportStatus, contextToken, checkedCanvas, cancelUpload: () => activeUpload?.abort() };
}
