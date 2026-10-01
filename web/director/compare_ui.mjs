import { filmClipTimes } from './film_ui.mjs';
import { createModalAccess } from './modal_ui.mjs';

export function comparisonWindow(entries) {
    if(!Array.isArray(entries)||entries.length!==2)throw Error('对比需要两版成片');
    const times=entries.map(filmClipTimes);
    return {times,duration:Math.min(...times.map(time=>time.duration))};
}

// A is the transport master. B follows time, never playback rate or audio mix.
export function createVersionComparison({root,versions,resultKey,videos,prepare,escape:esc}) {
    const dialog=document.createElement('dialog');dialog.className='w-compare-dialog';dialog.dataset.compareDialog='';
    dialog.innerHTML='<div class="o-row o-between"><h3>本镜 A/B 对比</h3><button type="button" data-compare-close>关闭</button></div><p>使用固定媒体副本，不修改采用版本。按相同播放时间比较共同长度，不保证动作逐帧对应；声音只听一路。</p><div class="w-compare-selectors"><label>A 版本<select data-compare-a></select></label><label>B 版本<select data-compare-b></select></label><button type="button" data-compare-prepare>准备对比</button><label>监听声音<select data-compare-audio><option value="a">仅 A</option><option value="b">仅 B</option></select></label></div><p data-compare-status role="status"></p><div class="w-compare-videos"><figure><figcaption>A · 主控播放</figcaption><video data-compare-video-a controls playsinline preload="metadata"></video></figure><figure><figcaption>B · 同步跟随</figcaption><video data-compare-video-b playsinline preload="metadata"></video></figure></div>';
    root.append(dialog);
    const modal=createModalAccess(dialog,{root});
    const $=selector=>dialog.querySelector(selector),a=$('[data-compare-video-a]'),b=$('[data-compare-video-b]'),status=$('[data-compare-status]');
    let sequence=0,window=null,timer=null,ready=false;
    const pause=()=>{a.pause();b.pause();};
    function audio(){const listen=$('[data-compare-audio]').value;a.muted=listen!=='a';b.muted=listen!=='b';}
    function clear(){sequence++;ready=false;window=null;pause();if(timer)clearInterval(timer);timer=null;}
    function close(){clear();dialog.close();}
    $('[data-compare-close]').addEventListener('click',close);
    dialog.addEventListener('cancel',clear);
    dialog.addEventListener('close',clear);
    $('[data-compare-audio]').addEventListener('change',audio);
    // Native video volume controls may unmute A even while the selector listens to B.
    // Honor that explicit control, but never leave both tracks audible.
    a.addEventListener('volumechange',()=>{if(!a.muted){if(!b.muted)b.muted=true;$('[data-compare-audio]').value='a';}});
    b.addEventListener('volumechange',()=>{if(!b.muted){if(!a.muted)a.muted=true;$('[data-compare-audio]').value='b';}});
    function sync(force=false){
        if(!ready||!window)return;
        const elapsed=Math.max(0,Math.min(window.duration,a.currentTime-window.times[0].start));
        const target=window.times[1].start+elapsed;
        if(force||Math.abs(b.currentTime-target)>0.08)b.currentTime=target;
        if(a.currentTime<window.times[0].start)a.currentTime=window.times[0].start;
        if(elapsed>=window.duration-0.005){pause();status.textContent='共同区间播放结束；可拖动 A 进度条重新查看。';}
    }
    a.addEventListener('play',()=>{
        if(!ready){pause();return;}
        if(a.currentTime-window.times[0].start>=window.duration-0.01)a.currentTime=window.times[0].start;
        sync(true);audio();const ticket=sequence;
        b.play().catch(()=>{if(ticket===sequence){pause();status.textContent='B 播放被阻止，请重新点击 A 播放。';}});
    });
    a.addEventListener('pause',()=>b.pause());
    a.addEventListener('seeking',()=>sync(true));
    a.addEventListener('ratechange',()=>{a.playbackRate=1;b.playbackRate=1;});
    a.addEventListener('ended',pause);b.addEventListener('ended',pause);
    for(const video of [a,b])video.addEventListener('error',()=>{if(dialog.open&&window){ready=false;pause();status.textContent='固定媒体无法播放；不会自动替换版本，请重新准备或检查原片。';}});
    for(const selector of ['[data-compare-a]','[data-compare-b]'])$(selector).addEventListener('change',()=>{clear();status.textContent='选择已更改，请点击“准备对比”。';});
    async function preparePair(){
        clear();const ticket=sequence;status.textContent='正在核对媒体并保留固定副本…';a.removeAttribute('src');b.removeAttribute('src');a.load();b.load();
        const keys=[$('[data-compare-a]').value,$('[data-compare-b]').value];
        try{
            if(keys[0]===keys[1])throw Error('请选择两个不同的版本');
            const entries=await prepare(keys);
            if(ticket!==sequence||!dialog.open)return;
            if(!entries){status.textContent='工程或镜头已变化，请关闭后重新对比。';return;}
            window=comparisonWindow(entries);audio();let loaded=0;
            [a,b].forEach((video,index)=>{
                video.onloadedmetadata=()=>{
                    if(ticket!==sequence||!dialog.open)return;
                    video.currentTime=window.times[index].start;
                    if(++loaded===2){ready=true;status.textContent=`已固定两版 · 共同区间 ${window.duration.toFixed(2)} 秒 · 点击 A 播放`;timer=setInterval(()=>sync(),50);}
                };
                video.src=entries[index].media_url;
            });
        }catch(error){if(ticket===sequence&&dialog.open)status.textContent='无法对比：'+error.message;}
    }
    $('[data-compare-prepare]').addEventListener('click',preparePair);
    function open(preferred){
        clear();root.querySelectorAll('video,audio').forEach(media=>media.pause());
        const records=versions().filter(row=>row.state==='success'&&videos(row).length===1);
        const options=records.map((row,index)=>`<option value="${esc(resultKey(row))}">${esc(`${records.length-index} · ${row.submitted_at?new Date(row.submitted_at*1000).toLocaleString():'时间未记录'} · seed ${row.seed??'未记录'}`)}</option>`).join('');
        $('[data-compare-a]').innerHTML=options;$('[data-compare-b]').innerHTML=options;
        const first=records.find(row=>resultKey(row)===preferred)||records[0],second=records.find(row=>row!==first);
        $('[data-compare-a]').value=first?resultKey(first):'';$('[data-compare-b]').value=second?resultKey(second):'';
        modal.show();
        if(records.length<2){status.textContent='至少需要本镜两个成功的单一成片版本。';return;}
        preparePair();
    }
    return {open};
}
