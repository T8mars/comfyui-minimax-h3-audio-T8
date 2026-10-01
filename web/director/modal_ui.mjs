// Only names and return focus are shared. Cancellation, IME and async work stay
// with each dialog's owner; this helper never intercepts Escape or closes media.
let titleSequence=0;

export function createModalAccess(dialog,{root=dialog.parentElement}={}) {
    let returnTargets=[];
    function name(){
        const heading=dialog.querySelector('h3,h2,h1');
        if(!heading)return;
        if(!heading.id||document.getElementById(heading.id)!==heading){
            let id;do{id='director-modal-title-'+(++titleSequence);}while(document.getElementById(id));
            heading.id=id;
        }
        dialog.setAttribute('aria-labelledby',heading.id);
    }
    function visible(element){
        return !!element?.isConnected&&!element.disabled&&!element.closest('[inert]')&&element.getClientRects().length>0&&getComputedStyle(element).visibility!=='hidden';
    }
    dialog.addEventListener('close',()=>{
        if(dialog.open)return;
        const targets=returnTargets;returnTargets=[];
        // Another dialog may have opened while this close event was queued.
        if(document.querySelector('dialog[open]'))return;
        // Respect an explicit navigation/editor focus made by the owner on close.
        const active=document.activeElement;
        if(active!==document.body&&active!==document.documentElement&&!dialog.contains(active)&&visible(active))return;
        const target=targets.find(visible);
        target?.focus({preventScroll:true});
    });
    name();
    return {show(opener=document.activeElement){
        name();
        if(dialog.open)return;
        returnTargets=[];
        // Menus fold after activation: remember their summaries, not a hidden child.
        let menu=opener?.closest?.('details');
        while(menu){const summary=menu.querySelector(':scope > summary');if(summary)returnTargets.push(summary);menu=menu.parentElement?.closest('details');}
        if(!returnTargets.length&&opener&&!dialog.contains(opener))returnTargets.push(opener);
        const fallback=root?.querySelector('.w-header-actions > .w-menu > summary');
        if(fallback&&!returnTargets.includes(fallback))returnTargets.push(fallback);
        dialog.showModal();
    }};
}
