// N04: two plain-text recipes for the existing snapshot/candidate workflow.
// No asset database, inferred evidence, timing transform, queue or provider.
const string = (value) => ({type: 'string', max_length: 4096, default: value});
const recipes = {
    continuity: {
        name: '角色道具连续账本 · 本镜声明', scope: 'shot',
        text: '[本镜连续性声明；不是自动识别或已确认事实]\n全片唯一事件/样本时间线：${global_timeline}\n稳定角色ID、服装与身份：${roles}\n本镜在场与离场：${presence}\n道具起始状态、持有人与允许转移：${props}\n承接的实际已采用末态：${previous_end_state}\n本镜允许的动作/状态变化：${allowed_changes}\n本镜必须保留的结束状态：${end_state}\n只按本镜明确声明的变化推进，不另造道具、不交换角色。未知状态明确保留未知；上述文字不自动成为sourceEvidence。需要跨镜事实时，在创作意图里显式绑定已确认镜头UUID/evidence SHA，不用partial x0或未采用候选充当实际成片末态。',
        parameter_schema: {
            global_timeline: string('待填：全片时间原点、单位/采样率与本镜起止；不重置局部时钟'),
            roles: string('待填：A/B稳定ID及实际参考包/素材对应关系，不猜身份'),
            presence: string('待填：本镜每个role_id的visual/voice声明'),
            props: string('待填：道具ID、持有人、位置与已知状态；无来源填未知'),
            previous_end_state: string('未知；待选择实际已采用成片/已确认末态，不填partial4'),
            allowed_changes: string('待填：本镜唯一允许的动作及道具转移'),
            end_state: string('待填：下一镜需要承接的角色/道具状态'),
        },
    },
    offscreen: {
        name: '画内/画外对话 · 本镜声明', scope: 'shot',
        text: '[本镜角色/声音声明；不是声纹锁定或硬时间隔离]\n全片唯一事件/样本时间线：${global_timeline}\n有序角色与参考包对应：${ordered_refs}\n本镜presence与声音锚：${role_plan}\n声音策略：${audio_policy}\n实际master与完成状态：${master}\n本镜对白事件（只写局部事件，不重复进global prompt）：${dialogue_events}\n人物离场时同时撤其Qwen视觉grounding和VAE视觉引用；明确画外说话人仍可保留voice anchor。A/B的音色reference_only不是drive_audio/final_audio，不互换编号。已有配音只用一个全片master驱动，完成源音可明确lock_source/final_audio；无配音保持原生联合生成，并使用既有1.92 Speech Scope/外置Relay事件时间线。不得冻结未完成LOW partial4音频，不自动TTS/ASR替换，不把字面规则当已修改节点接线。',
        parameter_schema: {
            global_timeline: string('待填：全片时间原点、单位/采样率与本镜起止；不改时钟'),
            ordered_refs: string('待填：按实际Route输入顺序填写包名/SHA与role_id；不自动加载'),
            role_plan: string('[{"role_id":"A","visual":false,"voice":true},{"role_id":"B","visual":true,"voice":false}]'),
            audio_policy: {type: 'string', choices: ['native_joint_exp', 'completed_master_lock_source'], default: 'native_joint_exp'},
            master: string('无既有配音；原生联合生成。若改master策略须填写实际来源与完成态回执'),
            dialogue_events: string('待填：说话人ID、全片起止/样本范围、各一次对白；不从观察事实猜台词'),
        },
    },
};

export function continuityRecipe(id) {
    if (!Object.hasOwn(recipes, id)) throw Error('未知连续性配方');
    return structuredClone({...recipes[id], parameters: {}});
}

export function fillContinuityRecipeForm(fields, id) {
    // A nonempty editor is user data. Never silently replace an existing draft.
    for (const key of ['name', 'text', 'parameter_schema', 'parameters']) {
        const value = fields[key].value.trim();
        if (value && !(key.includes('parameter') && value === '{}')) {
            throw Error('规则表单已有草稿；请先保留/添加它，再手动清空表单后填写配方。');
        }
    }
    const recipe = continuityRecipe(id);
    for (const key of ['name', 'text', 'scope']) fields[key].value = recipe[key];
    for (const key of ['parameter_schema', 'parameters']) fields[key].value = JSON.stringify(recipe[key], null, 2);
    return recipe.name;
}
