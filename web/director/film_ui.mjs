import { createModalAccess } from './modal_ui.mjs';

// Playlist playback is intentionally separate from encoding a deliverable MP4.
export function filmClipTimes(entry) {
    const fps=entry?.media?.fps;
    if(!Number.isFinite(fps)||fps<=0||!Number.isInteger(entry.in_frame)||!Number.isInteger(entry.out_frame)||entry.in_frame<0||entry.out_frame<=entry.in_frame)throw Error('整片镜头时间无效');
    const first=entry.origin==='external'?entry.decoded_clock?.video?.first_time:null;
    const offset=first?first.num/first.den:0;
    if(!Number.isFinite(offset))throw Error('外片源时钟无效');
    return {start:offset+entry.in_frame/fps,end:offset+entry.out_frame/fps,duration:(entry.out_frame-entry.in_frame)/fps};
}
export function createFilmViewer({root,prepare,notify,locate,escape:esc,startExport,exportStatus}) {
    const dialog=document.createElement('dialog');dialog.className='w-film-dialog';dialog.dataset.filmDialog='';
    dialog.innerHTML='<div class="o-row o-between"><h3>整片串播</h3><div class="o-row"><button type="button" data-film-last-export>上次导出</button><button type="button" data-film-close>关闭</button></div></div><p data-film-state role="status"></p><p data-film-export-status role="status"></p><div data-film-body></div>';
    root.append(dialog);
    const modal=createModalAccess(dialog,{root});
    let manifest=null,index=0,sequence=0,exportSequence=0,advancing=false,finished=false;
    const body=dialog.querySelector('[data-film-body]'),status=dialog.querySelector('[data-film-state]');
    function close(){sequence++;body.querySelectorAll('video').forEach(video=>video.pause());dialog.close();}
    dialog.querySelector('[data-film-close]').addEventListener('click',close);
    dialog.addEventListener('cancel',()=>{sequence++;body.querySelectorAll('video').forEach(video=>video.pause());});
    async function followExport(target,ticket,exportTicket){
        const host=dialog.querySelector('[data-film-export-status]');
        while(dialog.open&&ticket===sequence&&exportTicket===exportSequence){
            try{
                const job=await exportStatus(target);
                if(!dialog.open||ticket!==sequence||exportTicket!==exportSequence)return;
                if(!job){host.textContent='当前项目没有可读取的导出任务。';return;}
                target=job.target;
                if(job.state==='success'){
                    host.innerHTML=`导出已完成 · ${job.media.width}×${job.media.height} · ${job.media.frames} 帧 <a href="${esc(job.download_url)}" download>下载整片 MP4</a><small> · 使用该任务冻结的清单，不代表当前草稿。SHA ${esc(job.output_sha256)}</small>`;return;
                }
                if(!['queued','running'].includes(job.state)){host.textContent='导出未完成：'+(job.error||job.state)+'；原片保留，可明确重新导出。';return;}
                host.textContent=`CPU 导出中 · ${job.done}/${job.total} 镜。关闭窗口不会取消；可从“上次导出”恢复查看。`;
                await new Promise(resolve=>setTimeout(resolve,1000));
            }catch(error){if(ticket===sequence&&exportTicket===exportSequence)host.textContent='暂时无法核对导出：'+error.message+'；没有自动重发，请点“上次导出”重查。';return;}
        }
    }
    dialog.querySelector('[data-film-last-export]').addEventListener('click',()=>followExport(null,sequence,++exportSequence));
    function drawCurrent(autoplay=false){
        const player=body.querySelector('video'),entry=manifest.entries[index],times=filmClipTimes(entry),ticket=sequence;
        advancing=true;finished=false;
        const describe=()=>{status.textContent=`第 ${index+1}/${manifest.entries.length} 镜 · ${entry.name} · 本段 ${times.duration.toFixed(2)} 秒`;};
        const live=()=>ticket===sequence&&dialog.open;
        const lastFrame=times.end-1/entry.media.fps;
        describe();
        body.querySelectorAll('[data-film-shot]').forEach(button=>button.setAttribute('aria-current',String(Number(button.dataset.filmShot)===index)));
        player.onloadedmetadata=()=>{
            if(ticket!==sequence||!dialog.open)return;
            player.currentTime=times.start;advancing=false;
            if(autoplay)player.play().catch(()=>{status.textContent+=' · 请点击播放';});
        };
        player.onended=()=>next(true);
        player.onplay=()=>{
            if(!live()||advancing){player.pause();return;}
            // A native replay after the last clip is still a bounded clip play.
            // Seeking back inside the range first keeps the user's chosen position.
            if((finished&&player.currentTime>=lastFrame-0.001)||player.currentTime<times.start||player.currentTime>=times.end-0.5/entry.media.fps)player.currentTime=times.start;
            finished=false;describe();
        };
        player.ontimeupdate=()=>{if(live()&&!advancing&&!player.paused&&player.currentTime>=times.end-0.5/entry.media.fps)next(true);};
        player.onseeking=()=>{if(!live()||advancing)return;if(player.currentTime<times.start)player.currentTime=times.start;else if(player.currentTime>=times.end)player.currentTime=lastFrame;};
        player.onerror=()=>{advancing=true;status.textContent='此镜无法播放或冻结媒体校验失败；不会跳过或替换，请核对。';};
        player.src=entry.media_url;
    }
    function next(autoplay){
        if(advancing||finished||!dialog.open)return;
        const player=body.querySelector('video');player.pause();advancing=true;
        if(index+1<manifest.entries.length){index++;drawCurrent(autoplay);}
        else {
            const entry=manifest.entries[index],times=filmClipTimes(entry);
            finished=true;advancing=false;player.currentTime=times.end-1/entry.media.fps;
            status.textContent=`整片播放结束 · ${manifest.duration.toFixed(2)} 秒。串播可能有加载间隙，不是已导出的 MP4。`;
        }
    }
    dialog.addEventListener('click',event=>{
        const button=event.target.closest('button');if(!button)return;
        if(button.dataset.filmLocate){close();locate(button.dataset.filmLocate);}
        if(button.dataset.filmShot!==undefined){index=Number(button.dataset.filmShot);drawCurrent(false);}
        if(button.hasAttribute('data-film-previous')&&index>0){index--;drawCurrent(false);}
        if(button.hasAttribute('data-film-next')){advancing=false;next(false);}
        if(button.hasAttribute('data-film-restart')){index=0;drawCurrent(true);}
        if(button.hasAttribute('data-film-export')){
            const width=body.querySelector('[data-film-width]'),height=body.querySelector('[data-film-height]'),confirm=body.querySelector('[data-film-encoding]');
            if(!width.reportValidity()||!height.reportValidity()||!confirm.reportValidity())return;
            const ticket=sequence,exportTicket=++exportSequence;button.disabled=true;
            const options={width:width.valueAsNumber,height:height.valueAsNumber,policy:'h264_24fps_aac48k_stereo_contain_v1',confirm_encoding:confirm.checked,allow_silent:body.querySelector('[data-film-silence]').checked};
            dialog.querySelector('[data-film-export-status]').textContent='正在登记导出；不加载模型、不重新采样。';
            startExport(manifest,options).then(target=>followExport(target,ticket,exportTicket)).catch(error=>{if(ticket===sequence)dialog.querySelector('[data-film-export-status]').textContent='导出提交未确认：'+error.message+'；可先点“上次导出”核对，未自动重发。';}).finally(()=>{button.disabled=false;});
        }
    });
    async function open(){
        const ticket=++sequence;manifest=null;index=0;
        exportSequence++;dialog.querySelector('[data-film-export-status]').textContent='';
        root.querySelectorAll('video,audio').forEach(media=>media.pause());
        body.replaceChildren();status.textContent='正在核对采用版本、解码并固定媒体副本；不加载模型、不提交生成…';
        modal.show();
        try{
            const result=await prepare();
            if(ticket!==sequence||!dialog.open)return;
            if(!result){status.textContent='项目或草稿已经变化，请关闭后重新准备整片。';return;}
            if(!result.ready){
                status.textContent='整片尚未准备好；没有替换任何采用版本。';
                body.innerHTML='<ul class="w-film-issues">'+result.errors.map(issue=>`<li>第 ${issue.shot_number} 镜 · ${esc(issue.name)}：${esc(issue.message)} <button type="button" data-film-locate="${esc(issue.shot_id)}">去处理</button></li>`).join('')+'</ul>';
                return;
            }
            manifest=result;
            body.innerHTML='<p class="o-tip">使用准备时的镜头顺序、采用版本及帧范围，媒体副本已固定 SHA。只做串播，不保证无缝切换；重新编辑后需重新准备。</p><video class="w-film-player" controls preload="metadata" playsinline aria-label="整片串播播放器"></video><div class="o-row"><button type="button" data-film-previous>上一镜</button><button type="button" data-film-restart>从头播放</button><button type="button" data-film-next>下一镜</button><a data-film-manifest download>下载固定清单</a></div><ol class="w-film-shots">'+manifest.entries.map((entry,i)=>`<li><button type="button" data-film-shot="${i}">${i+1} · ${esc(entry.name)} <small>${entry.seconds.toFixed(2)} 秒</small></button></li>`).join('')+'</ol>';
            body.querySelector('[data-film-manifest]').href=result.manifest_url;
            const first=manifest.entries[0].media;
            body.insertAdjacentHTML('beforeend',`<details class="w-film-export"><summary>导出整片 MP4</summary><p class="o-tip">独立 CPU 编码：24fps H.264、48kHz 双声道 AAC。按固定帧范围硬切，等比缩放＋黑边；不裁切、不加转场或音量处理。原片保留。只支持 24fps 成片。</p><div class="o-grid"><label>输出宽度<input type="number" data-film-width value="${first.width}" min="32" max="4096" step="2" required></label><label>输出高度<input type="number" data-film-height value="${first.height}" min="32" max="4096" step="2" required></label></div><label class="o-check"><input type="checkbox" data-film-encoding required>我确认以上画面与音频编码规格</label><label class="o-check"><input type="checkbox" data-film-silence>允许为没有音轨的镜头补静音</label><button type="button" data-film-export>开始导出 MP4</button></details>`);
            drawCurrent(false);
        }catch(error){if(ticket===sequence&&dialog.open)status.textContent='整片准备失败：'+error.message;else notify('整片准备未完成：'+error.message);}
    }
    return {open,close};
}
