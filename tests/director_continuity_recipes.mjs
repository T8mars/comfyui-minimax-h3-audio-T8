import assert from 'node:assert/strict';
import test from 'node:test';
import {continuityRecipe, fillContinuityRecipeForm} from '../web/director/continuity_recipes.mjs';

const fields = () => Object.fromEntries(['name', 'text', 'scope', 'parameter_schema', 'parameters'].map(key => [key, {value: key.includes('parameter') ? '{}' : ''}]));
for (const id of ['continuity', 'offscreen']) {
    test(`${id} fills only an empty form, defaults to this shot and owns its data`, () => {
        const form = fields(), name = fillContinuityRecipeForm(form, id), recipe = continuityRecipe(id);
        assert.equal(form.name.value, name);
        assert.equal(form.text.value, recipe.text);
        assert.equal(form.scope.value, 'shot');
        assert.deepEqual(JSON.parse(form.parameter_schema.value), recipe.parameter_schema);
        assert.deepEqual(JSON.parse(form.parameters.value), {});
        recipe.parameter_schema.global_timeline.default = 'mutation';
        assert.notEqual(continuityRecipe(id).parameter_schema.global_timeline.default, 'mutation');
        const before = structuredClone(form);
        assert.throws(() => fillContinuityRecipeForm(form, id), /已有草稿/);
        assert.deepEqual(form, before);
    });
}
test('unknown IDs and any nonempty text/parameter editor remain untouched', () => {
    const form = fields(), before = structuredClone(form);
    assert.throws(() => fillContinuityRecipeForm(form, '__proto__'), /未知/);
    assert.deepEqual(form, before);
    for (const key of ['name', 'text', 'parameter_schema', 'parameters']) {
        const next = fields(); next[key].value = '${untrusted}';
        const previous = structuredClone(next);
        assert.throws(() => fillContinuityRecipeForm(next, 'continuity'), /已有草稿/);
        assert.deepEqual(next, previous);
    }
});
