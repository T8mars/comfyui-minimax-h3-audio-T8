import { createModalAccess } from './modal_ui.mjs';

export function bundleBytes(bytes){return bytes>=1024**3?(bytes/1024**3).toFixed(2)+' GiB':bytes>=1024**2?(bytes/1024**2).toFixed(2)+' MiB':(bytes/1024).toFixed(1)+' KiB';}

export function createBundleDialog({root,escape:esc,prepare,build,upload,apply,lastImport,notify}) {
    const dialog=document.createElement('dialog');dialog.className='w-bundle-dialog';dialog.dataset.bundleDialog='';
    dialog.innerHTML='<div class="o-row o-between"><h3>工程包 ZIP</h3><button type="button" data-bundle-close>关闭</button></div><p>将工程文字、输入素材及采用成片一起迁移。不会打包模型权重、其它历史、环境密钥或运行目录。模型需在目标机器另行安装。</p><div class="w-bundle-actions"><section><h4>导出当前工程</h4><label><input type="checkbox" data-bundle-results checked> 包含已采用成片</label><button type="button" data-bundle-prepare>检查清单与大小</button></section><section><h4>导入为新工程</h4><label>选择工程 ZIP<input type="file" accept=".zip" data-bundle-file></label><button type="button" data-bundle-last>恢复上次导入请求</button></section></div><p data-bundle-status role="status"></p><div data-bundle-body></div>';
    root.append(dialog);
    const modal=createModalAccess(dialog,{root});
    const $=selector=>dialog.querySelector(selector),status=$('[data-bundle-status]'),body=$('[data-bundle-body]');
    let sequence=0,controller=null,plan=null,kind=null;
    function reset(){sequence++;controller?.abort();controller=null;plan=null;kind=null;body.replaceChildren();}
    function close(){reset();dialog.close();}
    $('[data-bundle-close]').addEventListener('click',close);
    dialog.addEventListener('cancel',reset);dialog.addEventListener('close',reset);
    function showPlan(value,type){
        if(!value){status.textContent=type==='import'?'没有可恢复的导入清单，请选择工程 ZIP。':'当前草稿已变化，请重新检查。';return;}
        if(!value.ready){status.textContent='工程包尚未准备好；没有生成 ZIP。';body.innerHTML='<ul>'+value.errors.map(error=>`<li>${esc(error.name||'')}: ${esc(error.message)}</li>`).join('')+'</ul>';return;}
        plan=value;kind=type;
        status.textContent=`${value.title} · ${value.shots} 镜 · ${value.files.length} 个媒体文件 · ${bundleBytes(value.media_bytes)}（ZIP另含少量设置文件）`;
        body.innerHTML=`<p>${type==='export'?'这是当前草稿的固定清单；后续编辑不改变它。不含未采用的历史版本。':'文件及SHA已检查。确认后创建新工程、镜头和素材身份；当前工程不被替换，不自动生成视频。'}</p><div class="w-bundle-files"><table><thead><tr><th>类型</th><th>名称</th><th>大小</th></tr></thead><tbody>${value.files.map(row=>`<tr><td>${row.role==='asset'?'输入素材':'采用成片'}</td><td>${esc(row.name)}</td><td>${bundleBytes(row.size)}</td></tr>`).join('')}</tbody></table></div><p class="o-tip">包内文字、素材和原始媒体元数据可能含私人内容，分享前请检查。未知扩展字段不包含在此格式中，原项目和原 JSON 均保留。</p><button type="button" data-bundle-commit>${type==='export'?'确认清单并生成 ZIP':'确认导入为新工程'}</button>`;
    }
    async function run(getPlan,type,message){
        reset();const ticket=sequence;status.textContent=message;
        try{const value=await getPlan();if(ticket===sequence&&dialog.open){showPlan(value,type);const tip=body.querySelector('.o-tip');if(tip)tip.textContent+=' 成片保留完整源文件，包含裁剪范围外的画面。';}}
        catch(error){if(ticket===sequence&&dialog.open)status.textContent='未完成：'+error.message;}
    }
    $('[data-bundle-prepare]').addEventListener('click',()=>run(()=>prepare($('[data-bundle-results]').checked),'export','正在核对素材、固定采用成片并计算清单；不加载模型…'));
    $('[data-bundle-results]').addEventListener('change',()=>{reset();status.textContent='导出范围已变化，请重新检查清单。';});
    $('[data-bundle-last]').addEventListener('click',()=>run(lastImport,'import','正在读取上次导入；不会新建重复项目…'));
    $('[data-bundle-file]').addEventListener('change',()=>{
        const file=$('[data-bundle-file]').files[0];if(!file)return;
        run(()=>{controller=new AbortController();return upload(file,controller.signal);},'import','正在上传、校验并检查真实媒体；当前工程不变…');
        $('[data-bundle-file]').value='';
    });
    body.addEventListener('click',async event=>{
        const button=event.target.closest('button');if(!button)return;
        if(button.dataset.openProject){close();return;}
        if(!button.hasAttribute('data-bundle-commit')||!plan||button.disabled)return;
        const ticket=sequence,frozen=plan,operation=kind;button.disabled=true;
        status.textContent=operation==='export'?'正在生成 ZIP；原文件保持不变…':'正在注册副本与采用成片；原工程保持不变…';
        try{
            const result=await(operation==='export'?build(frozen):apply(frozen));
            if(ticket!==sequence||!dialog.open){notify(operation==='export'?'工程 ZIP 已生成。':'工程包已导入新项目，可从“打开项目”查找。');return;}
            status.textContent=operation==='export'?'ZIP 已生成，可以下载。':'新工程已保存，当前编辑页未替换。';
            body.innerHTML=operation==='export'?`<a href="${esc(result.download_url)}" download="director-project.zip">下载工程 ZIP · ${bundleBytes(result.bytes)}</a>`:`<button data-open-project="${esc(result.id)}">打开导入的新工程</button><p>原工程的采用成片已恢复；导入不包含历史生成配置快照。</p>`;
        }catch(error){if(ticket===sequence&&dialog.open)status.textContent='未确认完成：'+error.message+'；可以使用相同请求重试，不会覆盖原工程。';}
        finally{button.disabled=false;}
    });
    return {open(){reset();status.textContent='先检查清单再确认；不自动切换工程。';modal.show();}};
}
