import { createModalAccess } from './modal_ui.mjs';

// A deliberately explicit text format, not an AI parser or a generation request.
export function parseStoryboardText(source) {
    if(typeof source!=='string'||source.length>1024*1024)throw Error('文本上限为 1 Mi 字符，请分批导入。');
    const rows=[],errors=[];let active=null;
    const lines=source.replace(/^\uFEFF/,'').replace(/\r\n?/g,'\n').split('\n');
    function finish(){
        if(!active)return;
        while(active.lines.length&&!active.lines[0].trim())active.lines.shift();
        while(active.lines.length&&!active.lines.at(-1).trim())active.lines.pop();
        const raw=active.lines.join('\n'),references=[];
        if(!raw.trim())errors.push(`第 ${active.line} 行的镜头缺少提示词。`);
        const prompt=raw.replace(/@(image|video|audio)\d+\b|<(Picture|Video|Audio) \d+>/g,token=>{
            references.push(token);
            if(token.startsWith('@'))return '@missing_'+token.slice(1);
            const [,kind,index]=token.match(/<(Picture|Video|Audio) (\d+)>/);
            return '@missing_'+({Picture:'image',Video:'video',Audio:'audio'}[kind])+index;
        });
        rows.push({name:active.name,duration:active.duration,prompt,line:active.line,references:[...new Set(references)]});
        active=null;
    }
    lines.forEach((line,i)=>{
        if(/^\s*#\s+/.test(line)){
            finish();
            const header=line.match(/^\s*#\s+(.+?)\s*\|\s*(\d+(?:\.\d+)?)\s*(?:s|秒)\s*$/i);
            if(!header){errors.push(`第 ${i+1} 行标题格式不明确；请用“# 镜头名称 | 4秒”。`);return;}
            const name=header[1].trim(),duration=Number(header[2]);
            if(!name||name.length>200||!Number.isFinite(duration)||duration<0.001){errors.push(`第 ${i+1} 行名称须为 1–200 字，时长须为正数（至少 0.001 秒）。`);return;}
            active={name,duration,line:i+1,lines:[]};
        }else if(active)active.lines.push(line.replace(/^(\s*)\\(#\s+)/,'$1$2'));
        else if(line.trim())errors.push(`第 ${i+1} 行不属于任何镜头；请先写镜头标题。`);
    });
    finish();
    if(!rows.length&&!errors.length)errors.push('请粘贴至少一个镜头，或使用示例格式。');
    if(rows.length>200)errors.push('一次最多导入 200 镜，请拆分文本。');
    const total=rows.reduce((sum,row)=>sum+row.duration,0);
    if(!Number.isFinite(total))errors.push('时长超出可表示范围，请修正。');
    return {rows,errors,total,unbound:rows.reduce((sum,row)=>sum+row.references.length,0)};
}

export function createStoryboardDialog(root, ctx) {
    const dialog=document.createElement('dialog');dialog.className='w-storyboard-dialog';dialog.dataset.storyboardDialog='';
    dialog.setAttribute('aria-labelledby','director-storyboard-title');
    dialog.innerHTML='<header class="o-row o-between"><h3 id="director-storyboard-title">导入文本分镜</h3><button type="button" data-storyboard-close>取消</button></header><p class="o-tip">每镜以“# 镜头名称 | 4秒”开头，下面原样写完整提示词；以反斜杠开头的 \\# 可作为正文。只解析这一明确格式，不自动猜测或拆写台词。</p><label>分镜文本<textarea data-storyboard-text aria-label="分镜导入文本" spellcheck="false" placeholder="# 开场 | 4秒&#10;人物走近镜头，微笑。&#10;&#10;# 回答 | 3.5秒&#10;人物说：“你好。”"></textarea></label><div class="o-row"><label>本机 UTF-8 文本<input type="file" data-storyboard-file accept=".txt,.md,text/plain,text/markdown"></label><button type="button" data-storyboard-example>填入示例格式</button><button type="button" data-storyboard-preview>解析并预览</button></div><p data-storyboard-state role="status"></p><div data-storyboard-preview-body></div><label data-storyboard-ack-row hidden><input type="checkbox" data-storyboard-ack>我确认原文引用先标记为 missing，导入后重新绑定，不沿用其它项目的 @ 编号。</label><p class="o-tip">确认后追加到镜头列表末尾，使用新镜头身份，默认新手模式和模型生成声音，跟随本工程全片设置。不会复制当前镜头的首尾帧、驱动音频、独立采样或采用成片；不覆盖旧镜头、不提交生成，可一次撤销。实际时长/依赖仍须生成前预检。</p><button type="button" class="o-primary" data-storyboard-apply disabled>确认追加镜头</button>';
    root.append(dialog);
    const modal=createModalAccess(dialog,{root});
    const $=selector=>dialog.querySelector(selector),text=$('[data-storyboard-text]'),status=$('[data-storyboard-state]'),body=$('[data-storyboard-preview-body]');
    let preview=null,ticket=null,sequence=0;
    function invalidate(message='文本已修改，请重新解析预览。'){
        preview=null;ticket=null;body.replaceChildren();$('[data-storyboard-apply]').disabled=true;
        $('[data-storyboard-ack-row]').hidden=true;$('[data-storyboard-ack]').checked=false;status.textContent=message;
    }
    function open(){
        if(root.querySelector('dialog[open]')){ctx.notify('请先关闭当前窗口，再导入文本分镜。');return;}
        sequence++;invalidate('先预览，再确认；现有镜头不会被覆盖。');modal.show();text.focus();
    }
    function close(){sequence++;invalidate('');dialog.close();}
    $('[data-storyboard-close]').addEventListener('click',close);
    dialog.addEventListener('close',()=>{sequence++;invalidate('');});
    text.addEventListener('input',()=>{sequence++;invalidate();});
    $('[data-storyboard-example]').addEventListener('click',()=>{
        if(text.value.trim()&&!confirm('用示例替换当前导入文本？工程内容不会改变。'))return;
        text.value='# 开场 | 4秒\n人物走近镜头，微笑。\n\n# 回答 | 3.5秒\n人物说：“你好。”';sequence++;invalidate();text.focus();
    });
    $('[data-storyboard-file]').addEventListener('change',async()=>{
        const file=$('[data-storyboard-file]').files[0];$('[data-storyboard-file]').value='';if(!file)return;
        if(text.value.trim()&&!confirm('用文件内容替换当前导入文本？工程内容不会改变。'))return;
        const own=++sequence,origin=ctx.context();invalidate('正在本机读取文件，不上传…');
        try{
            if(file.size>1024*1024)throw Error('文件大于 1 MiB，请分批导入。');
            const value=new TextDecoder('utf-8',{fatal:true}).decode(await file.arrayBuffer());
            if(own!==sequence||!dialog.open||origin!==ctx.context())return;
            text.value=value;invalidate('文件已本机读取，请解析预览。');
        }catch(error){if(own===sequence&&dialog.open)status.textContent='无法读取：'+error.message+'。请使用 UTF-8 文本。';}
    });
    $('[data-storyboard-preview]').addEventListener('click',()=>{
        sequence++;invalidate('');
        try{
            const value=parseStoryboardText(text.value);
            if(value.errors.length){status.textContent='没有导入，请修正以下行：';const list=document.createElement('ul');for(const error of value.errors.slice(0,30)){const row=document.createElement('li');row.textContent=error;list.append(row);}body.append(list);return;}
            preview=value;ticket=ctx.snapshot();
            status.textContent=`将追加 ${value.rows.length} 镜 · 总时长 ${Number(value.total.toFixed(3))} 秒 · ${value.unbound} 个未绑定引用`;
            body.innerHTML=value.rows.map((row,i)=>`<article><h4>${i+1}. ${ctx.escape(row.name)} · ${row.duration} 秒</h4><pre>${ctx.escape(row.prompt)}</pre></article>`).join('');
            $('[data-storyboard-ack-row]').hidden=!value.unbound;$('[data-storyboard-apply]').disabled=!!value.unbound;
        }catch(error){status.textContent=error.message;}
    });
    $('[data-storyboard-ack]').addEventListener('change',()=>{$('[data-storyboard-apply]').disabled=!preview||(preview.unbound&&!$('[data-storyboard-ack]').checked);});
    $('[data-storyboard-apply]').addEventListener('click',()=>{
        if(!preview||ticket!==ctx.snapshot()){invalidate('工程已变化，请重新解析预览再确认。');return;}
        if(preview.unbound&&!$('[data-storyboard-ack]').checked)return;
        try{ctx.apply(preview.rows);close();}catch(error){status.textContent=error.message;}
    });
    return {open};
}
