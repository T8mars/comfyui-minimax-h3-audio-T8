// Optional planned-text sidecars. Never read prompts, adopt takes or encode media.
export function newCaptionDraft(film) {
    return {schema:'t8.director.caption_input.v1',film_sha256:film.sha256,tracks:[]};
}
export function validateCaptionDraft(film,input) {
    if(!input||input.schema!=='t8.director.caption_input.v1'||input.film_sha256!==film.sha256||!Array.isArray(input.tracks))throw Error('字幕稿不属于当前冻结整片；请使用它原来的固定清单');
    const entries=new Map(film.entries.map(entry=>[entry.shot_id,entry])),seen=new Set();
    for(const track of input.tracks){
        const entry=entries.get(track?.shot_id);
        if(!entry||seen.has(track.shot_id)||track.version_id!==entry.version_id||track.media_sha256!==entry.media_sha256)throw Error('字幕稿的镜头、采用版本或媒体 SHA 不一致');
        seen.add(track.shot_id);
        if(!['manual_planned_text','speech_planned_text'].includes(track.source?.kind)||!['frames','samples'].includes(track.clock?.unit)||!Array.isArray(track.cues))throw Error('字幕来源、时钟或 cues 列表无效');
        if(track.clock.unit==='samples'&&(!Number.isSafeInteger(track.clock.rate)||track.clock.rate<=0))throw Error('字幕采样率必须是正整数');
        for(const cue of track.cues){
            if(!Number.isSafeInteger(cue?.start)||!Number.isSafeInteger(cue?.end)||cue.start<0||cue.end<=cue.start||typeof cue.text!=='string'||cue.text.includes('\0'))throw Error('字幕起止坐标或正文无效；使用非负整数，结束须晚于开始');
        }
    }
    return structuredClone(input);
}
export function captionResultHTML(result,esc) {
    return `字幕已保存 · ${result.cues.length} 条 · <a href="${esc(result.srt_url)}" download>下载 SRT</a> · <a href="${esc(result.vtt_url)}" download>下载 VTT</a> · <a href="${esc(result.manifest_url)}" download>来源清单</a><small> · 绑定冻结整片 ${esc(result.film_id)}，不是当前可变草稿；计划文本，未核实实际发音和逐字对齐。</small>`;
}
export function createFilmCaptionEditor({host,film,save,importSpeech,escape:esc}) {
    let draft=newCaptionDraft(film),revision=0,disposed=false,busy=false,saved=JSON.stringify(draft),objectURL=null;
    const entries=new Map(film.entries.map(entry=>[entry.shot_id,entry]));
    host.innerHTML=`<summary>字幕 SRT / VTT（可选）</summary><p class="o-tip">仅使用你填写或导入的计划文本，不从提示词提取对白，不加载 ASR、不改视频和音轨。坐标为源片帧，结束帧不包含；保存时按采用范围裁切并排列到整片。裁半句与重叠需人工复核。</p><div class="o-grid"><label>镜头<select data-caption-shot>${film.entries.map((entry,i)=>`<option value="${esc(entry.shot_id)}">${i+1} · ${esc(entry.name)}</option>`).join('')}</select></label><p data-caption-range class="o-tip"></p><label>开始帧<input data-caption-start type="number" min="0" step="1" required></label><label>结束帧<input data-caption-end type="number" min="1" step="1" required></label></div><label>计划字幕文字<textarea data-caption-text rows="3" placeholder="填写对白或旁白；无台词的镜头留空"></textarea></label><div class="o-row"><button type="button" data-caption-add>添加字幕</button><button type="button" data-caption-clear>清除此镜字幕</button></div><ol data-caption-rows></ol><details><summary>导入字幕稿 / Speech 时间表</summary><p class="o-tip">字幕稿 JSON 必须对应同一冻结清单。Speech 时间表导入到上面选择的镜头，并替换该镜字幕；只是你声明音轨对应，不代表系统核实了实际发音。</p><label>导入字幕稿 JSON<input data-caption-import type="file" accept=".json,application/json"></label><label class="o-check"><input data-caption-confirm type="checkbox">我确认 Speech 时间表对应所选镜头当前采用的音轨</label><label>导入 Speech 时间表 JSON<input data-caption-speech type="file" accept=".json,application/json"></label></details><div class="o-row"><button type="button" data-caption-save>保存并生成 SRT / VTT</button><a data-caption-draft download="director-caption-draft.json">下载字幕稿 JSON</a></div><p data-caption-status role="status"></p><ul data-caption-notices></ul>`;
    const $=selector=>host.querySelector(selector),state=$('[data-caption-status]');
    const current=()=>entries.get($('[data-caption-shot]').value);
    function setRange(){const entry=current();$('[data-caption-start]').value=entry.in_frame;$('[data-caption-end]').value=entry.out_frame;$('[data-caption-range]').textContent=`采用源片帧 [${entry.in_frame}, ${entry.out_frame}) · ${entry.media.fps} fps`;$('[data-caption-confirm]').checked=false;}
    function pendingText(){return $('[data-caption-text]').value.trim().length>0;}
    function isDirty(){return JSON.stringify(draft)!==saved||pendingText();}
    function changed(){revision++;state.textContent='字幕稿已更改，尚未保存新附件。';render();}
    function render(){
        $('[data-caption-rows]').innerHTML=draft.tracks.flatMap((track,ti)=>track.cues.map((cue,ci)=>`<li><div class="o-row o-between"><small>${esc(entries.get(track.shot_id).name)} · ${esc(track.clock.unit==='frames'?'源片帧':'音频采样点')} [${cue.start}, ${cue.end}) · ${esc(track.source.kind==='speech_planned_text'?'Speech 计划稿':'手写计划稿')}</small><button type="button" data-caption-remove="${ti}:${ci}">删除此条</button></div><p class="w-caption-cue-text">${esc(cue.text)}</p></li>`)).join('');
        if(objectURL)URL.revokeObjectURL(objectURL);
        objectURL=URL.createObjectURL(new Blob([JSON.stringify(draft,null,2)],{type:'application/json'}));$('[data-caption-draft]').href=objectURL;
    }
    function editable(value){busy=value;host.querySelectorAll('button,input,select,textarea').forEach(control=>control.disabled=value);}
    function alive(ticket){return !disposed&&ticket===revision;}
    function replaceTrack(track){draft.tracks=draft.tracks.filter(row=>row.shot_id!==track.shot_id);draft.tracks.push(track);changed();}
    host.addEventListener('change',async event=>{
        if(event.target.matches('[data-caption-shot]')){setRange();return;}
        const file=event.target.files?.[0];if(!file)return;
        const speech=event.target.matches('[data-caption-speech]');
        if(!speech&&!event.target.matches('[data-caption-import]'))return;
        const entry=current(),ticket=revision,confirmed=$('[data-caption-confirm]').checked;
        event.target.value='';
        if(speech&&!confirmed){state.textContent='请先确认时间表对应所选镜头的采用音轨。';return;}
        if((speech?draft.tracks.some(track=>track.shot_id===entry.shot_id):isDirty())&&!confirm(speech?'替换所选镜头已有的字幕稿？其它镜头不变。':'用文件替换当前字幕稿？未保存内容会被替换。'))return;
        editable(true);
        try{
            const value=JSON.parse((await file.text()).replace(/^\uFEFF/,''));if(!alive(ticket))return;
            if(speech){
                const track=await importSpeech(film,entry.shot_id,value,confirmed);
                if(!alive(ticket))return;
                if(!track){state.textContent='项目已切换，未把 Speech 时间表写入当前窗口；原稿保留。';return;}
                validateCaptionDraft(film,{...draft,tracks:[...draft.tracks.filter(row=>row.shot_id!==entry.shot_id),track]});replaceTrack(track);
            }else{draft=validateCaptionDraft(film,value);changed();}
            state.textContent='已导入计划稿；尚未保存，且未核实实际发音。';
        }catch(error){if(alive(ticket))state.textContent='导入失败，原稿保留：'+error.message;}
        finally{if(!disposed)editable(false);}
    });
    $('[data-caption-draft]').addEventListener('click',event=>{
        if(pendingText()){event.preventDefault();state.textContent='文字框内还有未添加的字幕，请先点“添加字幕”，再下载完整字幕稿。';$('[data-caption-text]').focus();}
    });
    host.addEventListener('click',async event=>{
        const button=event.target.closest('button');if(!button||busy||disposed)return;
        const entry=current();
        if(button.hasAttribute('data-caption-add')){
            const start=$('[data-caption-start]'),end=$('[data-caption-end]'),text=$('[data-caption-text]');
            if(!start.reportValidity()||!end.reportValidity())return;
            if(!text.value.trim()){state.textContent='请填写字幕文字；无台词的镜头无需添加。';text.focus();return;}
            let track=draft.tracks.find(row=>row.shot_id===entry.shot_id);
            if(track&&(track.clock.unit!=='frames'||track.source.kind!=='manual_planned_text')){state.textContent='该镜已导入 Speech 时间表。要改为手写帧字幕，请先清除此镜字幕；可先下载原稿备份。';return;}
            const cue={start:start.valueAsNumber,end:end.valueAsNumber,text:text.value};
            const candidate=track?{...track,cues:[...track.cues,cue]}:{shot_id:entry.shot_id,version_id:entry.version_id,media_sha256:entry.media_sha256,clock:{unit:'frames'},source:{kind:'manual_planned_text'},cues:[cue]};
            try{validateCaptionDraft(film,{...draft,tracks:[...draft.tracks.filter(row=>row.shot_id!==entry.shot_id),candidate]});replaceTrack(candidate);text.value='';}
            catch(error){state.textContent=error.message;}
        }
        if(button.hasAttribute('data-caption-clear')){
            if(draft.tracks.some(row=>row.shot_id===entry.shot_id)&&confirm('清除所选镜头的字幕稿？不会删除已保存的字幕附件或原片。')){draft.tracks=draft.tracks.filter(row=>row.shot_id!==entry.shot_id);changed();}
        }
        if(button.dataset.captionRemove){
            const [ti,ci]=button.dataset.captionRemove.split(':').map(Number);draft.tracks[ti].cues.splice(ci,1);draft.tracks=draft.tracks.filter(track=>track.cues.length);changed();
        }
        if(button.hasAttribute('data-caption-save')){
            if(pendingText()){state.textContent='文字框内还有未添加的字幕，请先点“添加字幕”。';return;}
            const ticket=revision,input=validateCaptionDraft(film,draft);editable(true);state.textContent='正在校验固定成片并生成字幕；不改媒体。';
            try{
                const result=await save(film,input);if(!alive(ticket))return;
                if(!result){state.textContent='项目已切换，未把旧项目的字幕结果写入当前窗口；请回原项目点“上次字幕”核对。';return;}
                saved=JSON.stringify(input);state.innerHTML=captionResultHTML(result,esc);
                const reasons={empty_text_omitted:'空文字已忽略',outside_adopted_trim_omitted:'字幕不在采用范围内，已忽略',below_millisecond_resolution_omitted:'字幕短于毫秒精度，已忽略',partially_trimmed_text_needs_review:'字幕被剪辑范围截断，保留全文，请复核',overlapping_cues_retained:'保留了重叠字幕，请复核'};
                $('[data-caption-notices]').innerHTML=result.notices.map(notice=>`<li>${esc(entries.get(notice.shot_id)?.name||notice.shot_id)} · ${notice.cue_index+1}：${esc(reasons[notice.reason]||notice.reason)}</li>`).join('');
            }catch(error){if(alive(ticket))state.textContent='字幕未确认保存：'+error.message+'。原视频和 MP4 导出状态不变；可用相同稿件重试，或查“上次字幕”。';}
            finally{if(!disposed)editable(false);}
        }
    });
    setRange();render();
    return {isDirty,dispose(){disposed=true;revision++;if(objectURL)URL.revokeObjectURL(objectURL);}};
}
