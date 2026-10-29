// Optional text sidecars. Actual recognition is an explicit, cancelable action.
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
        if(!['manual_planned_text','speech_planned_text','recognized_audio'].includes(track.source?.kind)||!['frames','samples'].includes(track.clock?.unit)||!Array.isArray(track.cues))throw Error('字幕来源、时钟或 cues 列表无效');
        if(track.clock.unit==='samples'&&(!Number.isSafeInteger(track.clock.rate)||track.clock.rate<=0))throw Error('字幕采样率必须是正整数');
        for(const cue of track.cues){
            if(!Number.isSafeInteger(cue?.start)||!Number.isSafeInteger(cue?.end)||cue.start<0||cue.end<=cue.start||typeof cue.text!=='string'||cue.text.includes('\0'))throw Error('字幕起止坐标或正文无效；使用非负整数，结束须晚于开始');
        }
    }
    return structuredClone(input);
}
export function captionResultHTML(result,esc) {
    const provenance=result.text_status?.startsWith('recognized_audio')?'含实际音轨识别草稿，可人工修改；未认证说话人和口型。':'计划文本，未核实实际发音和逐字对齐。';
    return `字幕已保存 · ${result.cues.length} 条 · <a href="${esc(result.srt_url)}" download>下载 SRT</a> · <a href="${esc(result.vtt_url)}" download>下载 VTT</a> · <a href="${esc(result.manifest_url)}" download>来源清单</a><small> · 绑定冻结整片 ${esc(result.film_id)}，不是当前可变草稿；${provenance}</small>`;
}
export function createFilmCaptionEditor({host,film,save,importSpeech,asr,escape:esc}) {
    let draft=newCaptionDraft(film),revision=0,disposed=false,busy=false,saved=JSON.stringify(draft),objectURL=null;
    let asrTarget=null,asrRunning=false;
    const entries=new Map(film.entries.map(entry=>[entry.shot_id,entry]));
    host.innerHTML=`<summary>字幕 SRT / VTT（可选）</summary><p class="o-tip">仅使用你填写或导入的计划文本，不从提示词提取对白，不加载 ASR、不改视频和音轨。坐标为源片帧，结束帧不包含；保存时按采用范围裁切并排列到整片。裁半句与重叠需人工复核。</p><div class="o-grid"><label>镜头<select data-caption-shot>${film.entries.map((entry,i)=>`<option value="${esc(entry.shot_id)}">${i+1} · ${esc(entry.name)}</option>`).join('')}</select></label><p data-caption-range class="o-tip"></p><label>开始帧<input data-caption-start type="number" min="0" step="1" required></label><label>结束帧<input data-caption-end type="number" min="1" step="1" required></label></div><label>计划字幕文字<textarea data-caption-text rows="3" placeholder="填写对白或旁白；无台词的镜头留空"></textarea></label><div class="o-row"><button type="button" data-caption-add>添加字幕</button><button type="button" data-caption-clear>清除此镜字幕</button></div><ol data-caption-rows></ol><details data-caption-import-panel><summary>导入字幕稿 / Speech 时间表</summary><p class="o-tip">字幕稿 JSON 必须对应同一冻结清单。Speech 时间表导入到上面选择的镜头，并替换该镜字幕；只是你声明音轨对应，不代表系统核实了实际发音。</p><label>导入字幕稿 JSON<input data-caption-import type="file" accept=".json,application/json"></label><label class="o-check"><input data-caption-confirm type="checkbox">我确认 Speech 时间表对应所选镜头当前采用的音轨</label><label>导入 Speech 时间表 JSON<input data-caption-speech type="file" accept=".json,application/json"></label></details><div class="o-row"><button type="button" data-caption-save>保存并生成 SRT / VTT</button><a data-caption-draft download="director-caption-draft.json">下载字幕稿 JSON</a></div><p data-caption-status role="status"></p><ul data-caption-notices></ul>`;
    const $=selector=>host.querySelector(selector),state=$('[data-caption-status]');
    if(asr){
        host.querySelector('.o-tip').textContent='手写和导入计划稿不会加载识别模型；也可从下方明确识别已采用音轨。字幕与原片分开保存，裁半句与重叠需人工复核。';
        host.querySelector('summary').insertAdjacentHTML('afterend',`<details data-caption-asr><summary>从已采用音轨识别字幕</summary><p class="o-tip">使用本地模型在 CPU 识别所选镜头的真实音轨。结果先成为可编辑草稿；不改原片，不自动保存。词时间基于源片起点，采用范围只在导出时裁切一次；说话人和口型需自行核对。</p><label>本地识别模型<input data-caption-asr-model list="caption-asr-models" placeholder="选择已安装模型或填写本地目录"></label><datalist id="caption-asr-models"></datalist><label>语言<select data-caption-asr-language><option value="auto">自动识别</option><option value="Chinese">中文</option><option value="English">英语</option></select></label><div class="o-row"><button type="button" data-caption-asr-start>识别所选镜头音轨</button><button type="button" data-caption-asr-last>读取上次识别</button><button type="button" data-caption-asr-cancel disabled>取消识别</button></div><p data-caption-asr-status role="status"></p></details>`);
        Promise.resolve(asr.models()).then(models=>{if(disposed||!Array.isArray(models))return;$('datalist').innerHTML=models.map(m=>`<option value="${esc(m.directory)}">${esc(m.name)}</option>`).join('');if(!$('[data-caption-asr-model]').value)$('[data-caption-asr-model]').value=models[0]?.directory||'';}).catch(error=>{if(!disposed)$('[data-caption-asr-status]').textContent='模型列表未读取，可填写本地目录：'+error.message;});
    }
    const current=()=>entries.get($('[data-caption-shot]').value);
    function setRange(){const entry=current();$('[data-caption-start]').value=entry.in_frame;$('[data-caption-end]').value=entry.out_frame;$('[data-caption-range]').textContent=`采用源片帧 [${entry.in_frame}, ${entry.out_frame}) · ${entry.media.fps} fps`;$('[data-caption-confirm]').checked=false;}
    function pendingText(){return $('[data-caption-text]').value.trim().length>0||!!$('[data-caption-edit-panel]');}
    function isDirty(){return JSON.stringify(draft)!==saved||pendingText();}
    function changed(){revision++;state.textContent='字幕稿已更改，尚未保存新附件。';render();}
    function render(){
        $('[data-caption-rows]').innerHTML=draft.tracks.flatMap((track,ti)=>track.cues.map((cue,ci)=>`<li><div class="o-row o-between"><small>${esc(entries.get(track.shot_id).name)} · ${esc(track.clock.unit==='frames'?'源片帧':'音频采样点')} [${cue.start}, ${cue.end}) · ${esc(track.source.kind==='recognized_audio'?'音轨识别草稿':track.source.kind==='speech_planned_text'?'Speech 计划稿':'手写计划稿')}</small><button type="button" data-caption-edit="${ti}:${ci}">编辑</button><button type="button" data-caption-remove="${ti}:${ci}">删除此条</button></div><p class="w-caption-cue-text">${esc(cue.text)}</p></li>`)).join('');
        if(objectURL)URL.revokeObjectURL(objectURL);
        objectURL=URL.createObjectURL(new Blob([JSON.stringify(draft,null,2)],{type:'application/json'}));$('[data-caption-draft]').href=objectURL;
    }
    function editable(value){busy=value;host.querySelectorAll('button,input,select,textarea').forEach(control=>control.disabled=value);if(asr)$('[data-caption-asr-cancel]').disabled=!asrRunning||!asrTarget;}
    function alive(ticket){return !disposed&&ticket===revision;}
    function replaceTrack(track){draft.tracks=draft.tracks.filter(row=>row.shot_id!==track.shot_id);draft.tracks.push(track);changed();}
    async function followASR(job,ticket){
        const status=$('[data-caption-asr-status]');
        while(alive(ticket)&&job){
            asrTarget=job.target;editable(true);
            if(job.state==='success'){
                if(job.film_id!==film.id||job.request?.film_sha256!==film.sha256||job.track?.shot_id!==current().shot_id){status.textContent='识别完成，但对应另一份冻结清单或镜头，未套用到这里。请选择原镜头和固定清单核对。';return;}
                validateCaptionDraft(film,{...draft,tracks:[...draft.tracks.filter(t=>t.shot_id!==job.track.shot_id),job.track]});
                replaceTrack(job.track);status.textContent=`识别草稿已加入 ${job.track.cues.length} 条；请编辑文字和时间，再保存字幕。${job.untimed_words?.length?' 以下内容缺少词时间，请手动添加：'+job.untimed_words.map(w=>w.word).join('；'):''}`;return;
            }
            if(!['queued','running'].includes(job.state)){status.textContent='识别未完成：'+(job.error||job.state)+'。原片和字幕稿保留；没有自动重跑。';return;}
            status.textContent=job.cancel_requested?'正在结束本次识别工作进程…':'正在识别真实音轨，可取消；不修改视频。';
            await new Promise(resolve=>setTimeout(resolve,400));if(!alive(ticket))return;
            job=await asr.status(asrTarget);
        }
        if(alive(ticket)&&!job)status.textContent='项目已切换，未把旧识别结果写入当前窗口。';
    }
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
        const button=event.target.closest('button');if(!button||disposed)return;
        if(button.hasAttribute('data-caption-asr-cancel')){if(asrTarget){button.disabled=true;try{await asr.cancel(asrTarget);$('[data-caption-asr-status]').textContent='已请求取消，等待确认工作进程退出。';}catch(error){$('[data-caption-asr-status]').textContent='未确认取消：'+error.message;}}return;}
        if(busy)return;
        if($('[data-caption-edit-panel]')&&!button.dataset.captionEditApply&&!button.hasAttribute('data-caption-edit-cancel')){state.textContent='请先应用或取消正在编辑的字幕，未丢弃编辑内容。';return;}
        const entry=current();
        if(asr&&(button.hasAttribute('data-caption-asr-start')||button.hasAttribute('data-caption-asr-last'))){
            if(pendingText()){state.textContent='请先添加文字框里的字幕或应用正在编辑的内容。';return;}
            if(draft.tracks.some(t=>t.shot_id===entry.shot_id)&&!confirm('识别完成后替换此镜头的字幕草稿？其他镜头和原片不变。'))return;
            const ticket=revision;asrRunning=true;asrTarget=null;editable(true);
            try{const job=button.hasAttribute('data-caption-asr-last')?await asr.last(film):await asr.start(film,entry.shot_id,{model_directory:$('[data-caption-asr-model]').value,language:$('[data-caption-asr-language]').value});if(!job){$('[data-caption-asr-status]').textContent='没有可读取的识别任务，原稿保留。';return;}await followASR(job,ticket);}
            catch(error){if(alive(ticket))$('[data-caption-asr-status]').textContent='未确认识别完成：'+error.message+'。可读取上次识别核对，不会自动重发。';}
            finally{asrRunning=false;if(!disposed)editable(false);}return;
        }
        if(button.dataset.captionEdit){
            const [ti,ci]=button.dataset.captionEdit.split(':').map(Number),track=draft.tracks[ti],cue=track.cues[ci],rate=track.clock.unit==='frames'?entries.get(track.shot_id).media.fps:track.clock.rate;
            host.querySelectorAll('[data-caption-edit-panel]').forEach(panel=>panel.remove());
            button.closest('li').insertAdjacentHTML('beforeend',`<div data-caption-edit-panel><label>开始 / 秒<input data-caption-edit-start type="number" min="0" step="any" value="${cue.start/rate}"></label><label>结束 / 秒<input data-caption-edit-end type="number" min="0" step="any" value="${cue.end/rate}"></label><textarea data-caption-edit-text rows="2">${esc(cue.text)}</textarea><button type="button" data-caption-edit-apply="${ti}:${ci}">应用修改</button><button type="button" data-caption-edit-cancel>取消修改</button></div>`);return;
        }
        if(button.hasAttribute('data-caption-edit-cancel')){button.closest('[data-caption-edit-panel]').remove();return;}
        if(button.dataset.captionEditApply){
            const [ti,ci]=button.dataset.captionEditApply.split(':').map(Number),track=draft.tracks[ti],rate=track.clock.unit==='frames'?entries.get(track.shot_id).media.fps:track.clock.rate;
            const next=structuredClone(draft);next.tracks[ti].cues[ci]={start:Math.round($('[data-caption-edit-start]').valueAsNumber*rate),end:Math.round($('[data-caption-edit-end]').valueAsNumber*rate),text:$('[data-caption-edit-text]').value};
            try{draft=validateCaptionDraft(film,next);changed();}catch(error){state.textContent=error.message;}return;
        }
        if(button.hasAttribute('data-caption-add')){
            const start=$('[data-caption-start]'),end=$('[data-caption-end]'),text=$('[data-caption-text]');
            if(!start.reportValidity()||!end.reportValidity())return;
            if(!text.value.trim()){state.textContent='请填写字幕文字；无台词的镜头无需添加。';text.focus();return;}
            let track=draft.tracks.find(row=>row.shot_id===entry.shot_id);
            if(track&&track.source.kind!=='recognized_audio'&&(track.clock.unit!=='frames'||track.source.kind!=='manual_planned_text')){state.textContent='该镜已导入 Speech 时间表。要改为手写帧字幕，请先清除此镜字幕；可先下载原稿备份。';return;}
            const factor=track?.source.kind==='recognized_audio'?track.clock.rate/entry.media.fps:1;
            const cue={start:Math.round(start.valueAsNumber*factor),end:Math.round(end.valueAsNumber*factor),text:text.value};
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
