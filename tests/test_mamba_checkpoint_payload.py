"""Replay real cache owners with token-count payloads; no model or GPU execution."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import importlib.util
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS


def execute(nodes, namespace):
    module = ast.Module(body=nodes, type_ignores=[])
    exec("from __future__ import annotations\n" + ast.unparse(module), namespace)


def function(path, name):
    node = next(n for n in ast.walk(ast.parse(path.read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    return node


def replay(source, harness_path, patch, mode, tokens=229000, budget=2041, prior=None):
    import torch

    if prior is None:
        spec = importlib.util.spec_from_file_location("cache_harness", harness_path)
        harness = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = harness
        spec.loader.exec_module(harness)
        owners = harness.load_owners(source / 'v1/core')
        cache = harness.DraftCache(owners, 57344, draft_block_size=896)
        payload = torch.full((cache.pool.num_gpu_blocks, 1), -1, dtype=torch.int64)
    else:
        harness, owners, cache, payload = prior.harness, prior.owners, prior.cache, prior.payload
    request = harness.request(f"payload-{tokens}-{budget}", tokens=tokens,
        original=prior.request if prior else None,
        common_tokens=prior.request.num_prompt_tokens if prior else 0)
    request.num_tokens = request.num_prompt_tokens
    request.num_computed_tokens = cache.start(request)
    resumed_tokens = request.num_computed_tokens
    worker_indices = {}

    def copy_payload(bufs, config, funcs, gids, prev, current, bias, req_state, context):
        assert bias == 0
        src = req_state.block_ids[2][prev]
        dst = req_state.block_ids[2][current]
        payload[dst] = payload[src]

    pre_ns = dict(cdiv=lambda a, b: -(-a // b), itertools=itertools,
                  _resolve_fused_precopy=lambda ctx: None,
                  collect_mamba_copy_meta=copy_payload,
                  do_mamba_copy_block=lambda bufs: None)
    worker_source = source / 'v1/worker/mamba_utils.py'
    execute([function(worker_source, 'cleanup_mamba_state_idx'),
             function(worker_source, 'preprocess_mamba')], pre_ns)
    v2 = function(worker_source, 'preprocess_mamba_align_fused_kernel')
    v2_index = next(n.value for n in v2.body if isinstance(n, ast.Assign)
                    and isinstance(n.targets[0], ast.Name)
                    and n.targets[0].id == 'new_state_idx')
    v2_index = compile(ast.Expression(v2_index), str(worker_source), 'eval')

    metadata_ns = dict(torch=torch, MambaSpec=harness.MambaSpec)
    execute([function(source / 'v1/attention/backends/utils.py',
                      'mamba_get_block_table_tensor')], metadata_ns)
    builder = function(source / 'v1/attention/backends/gdn_attn.py', 'build')
    builder_end = next(i for i, n in enumerate(builder.body)
                       if isinstance(n, ast.If)
                       and ast.unparse(n.test) == 'spec_sequence_masks is None')
    builder_body = builder.body[:builder_end + 1]
    kda = function(source / 'models/glm5next/nvidia/kda.py', '_forward')
    prefill = next(n for n in ast.walk(kda) if isinstance(n, ast.If)
                   and any(isinstance(s, ast.Assign)
                           and isinstance(s.value, ast.Call)
                           and isinstance(s.value.func, ast.Name)
                           and s.value.func.id == 'gather_initial_states' for s in n.body))

    scheduler_source = patch.prepare(
        (source / patch.SCHEDULER).read_text(), patch.SCHEDULER)
    if mode in ('hash64', 'native896'):
        scheduler_source = scheduler_source.replace(patch.STOPS_NEW, patch.STOPS_OLD)
    if mode == 'native896':
        scheduler_source = scheduler_source.replace(patch.SPLIT_NEW, patch.SPLIT_OLD)
    scheduler_fn = next(n for n in ast.walk(ast.parse(scheduler_source))
                        if isinstance(n, ast.FunctionDef)
                        and n.name == '_mamba_block_aligned_split')
    scheduler_ns = {}
    execute([scheduler_fn], scheduler_ns)
    scheduler = NS(cache_config=NS(block_size=896, mamba_block_size=3584),
                   block_size=3584, hash_block_size=64, use_eagle=True,
                   max_num_scheduled_tokens=2048, mamba_partial_cache_hit=False,
                   scheduler_config=NS(long_prefill_token_threshold=0),
                   _glm53_align_prefill_limit=budget if budget < 2041 else None)
    setup = [n for n in ast.walk(ast.parse(scheduler_source))
             if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Attribute)
             and n.targets[0].attr in ('mamba_block_sizes', 'has_mamba_layers')]
    groups = [NS(kv_cache_spec=m.kv_cache_spec) for m in cache.managers]
    execute(setup, dict(self=scheduler, MambaSpec=harness.MambaSpec,
                        kv_cache_config=NS(kv_cache_groups=groups)))
    assert scheduler.mamba_block_sizes == [3584]
    failures = []
    seen = set()
    chunks = []
    boundary_trace = None
    while request.num_computed_tokens < request.num_prompt_tokens:
        start = request.num_computed_tokens
        count = scheduler_ns[scheduler_fn.name](scheduler, request,
                    min(budget, request.num_prompt_tokens - start))
        assert count > 0
        end = start + count
        chunks.append(count)
        for m in cache.managers:
            m.new_step_starts()
        for gid in cache.free_order:
            committed = max(0, start - 2048) if gid == 6 else start
            cache.managers[gid].remove_skipped_blocks(request.request_id, committed,
                                                     request.num_prompt_tokens)
        for m in cache.managers:
            m.allocate_new_blocks(request.request_id, end, end)
        block_ids = [[block.block_id for block in m.req_to_blocks[request.request_id]]
                     for m in cache.managers]
        output = NS(num_scheduled_tokens={request.request_id: count},
                    finished_req_ids=set(), preempted_req_ids=set(),
                    scheduled_cached_reqs=NS(resumed_req_ids=set()))
        req_state = NS(num_computed_tokens=start, block_ids=block_ids)
        pre_ns['preprocess_mamba'](output, None, NS(enable_prefix_caching=True),
            worker_indices, NS(req_ids=[request.request_id], num_accepted_tokens_cpu=[1]),
            {request.request_id: req_state}, {}, (),
            NS(mamba_group_ids=[2], mamba_spec=harness.MambaSpec(), offset=0))
        v2_column = eval(v2_index, {}, dict(computed_after=end, MAMBA_BLOCK_SIZE=3584))
        assert worker_indices[request.request_id] == v2_column
        query_start = torch.tensor([0, count], dtype=torch.int32)
        common = NS(query_start_loc=query_start, query_start_loc_cpu=query_start,
                    block_table_tensor=torch.tensor([block_ids[2]], dtype=torch.int32),
                    seq_lens=torch.tensor([end], dtype=torch.int32))
        builder_ns = dict(metadata_ns, common_attn_metadata=common,
            self=NS(kv_cache_spec=harness.MambaSpec(), use_spec_decode=True,
                    vllm_config=NS(cache_config=NS(mamba_cache_mode='align'))),
            num_decode_draft_tokens_cpu=None,
            split_decodes_and_prefills=lambda m, decode_threshold: (0, 1, 0, count))
        execute(builder_body, builder_ns)
        indices = builder_ns['non_spec_state_indices_tensor']
        assert int(indices[0]) == block_ids[2][v2_column]

        def chunk_payload(**kwargs):
            assert kwargs['output_final_state']
            assert kwargs['cu_seqlens'].tolist() == [0, count]
            assert int(kwargs['initial_state'][0, 0]) == start
            return None, kwargs['initial_state'] + count

        token = torch.zeros((1, 1, 1, 1))
        kda_ns = dict(q_ns=token, k_ns=token, v_ns=token, g1_ns=token, beta_ns=token,
            recurrent_state=payload, non_spec_state_indices_tensor=indices,
            has_initial_state=torch.tensor([start > 0]),
            non_spec_query_start_loc=builder_ns['non_spec_query_start_loc'],
            self=NS(A_log=None, dt_bias=None), safe_gate=True, lower_bound=-5,
            _rearr=lambda x: x, _cast_sigmoid=lambda x: x,
            gather_initial_states=lambda state, idx, ready: torch.where(
                ready[:, None], state[idx], torch.zeros_like(state[idx])),
            chunk_kda_with_fused_gate=chunk_payload,
            scatter_states=lambda state, src, idx: state.__setitem__(idx, src))
        execute(prefill.body, kda_ns)
        owners['hybrid_cache_blocks'](cache.coordinator, request, end)
        for boundary in range(3584, end + 1, 3584):
            if boundary in seen:
                continue
            hit = cache.pool.get_cached_block(request.block_hashes[boundary // 64 - 1], [2])
            if hit is None:
                continue
            seen.add(boundary)
            block = hit[0]
            stored = int(payload[block.block_id, 0])
            record = dict(boundary=boundary, payload_tokens=stored,
                          registered_after=end, chunk_start=start,
                          physical_block=block.block_id)
            if stored != boundary:
                failures.append(record)
            if boundary == 57344:
                boundary_trace = record
        request.num_computed_tokens = end
    result = dict(mode=mode, prompt_tokens=tokens, resumed_tokens=resumed_tokens,
                budget=budget, steps=len(chunks), tokens=sum(chunks),
                chunk_histogram=dict(Counter(chunks)),
                non_kpool_aligned_chunk_ends=sum((resumed_tokens + sum(chunks[:i+1])) % 4 != 0
                                                for i in range(len(chunks)-1)),
                checked_checkpoints=len(seen), payload_failures=failures,
                boundary_57344=boundary_trace)
    cache.finish(request)
    return result, NS(harness=harness, owners=owners, cache=cache,
                      payload=payload, request=request)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    recipe = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "memory_patch", recipe / 'overlay/patch_hybrid_memory_budget.py')
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    harness = recipe / 'tests/test_apc_periodic_retention.py'
    results = []
    for tokens in (229000, 229768, 232000):
        broken, _ = replay(args.source_root, harness, patch, 'hash64', tokens=tokens)
        fixed, _ = replay(args.source_root, harness, patch, 'checkpoint', tokens=tokens)
        assert broken['payload_failures']
        assert fixed['checked_checkpoints'] >= 5
        assert not fixed['payload_failures']
        assert fixed['non_kpool_aligned_chunk_ends'] == 0
        results.extend([broken, fixed])
    native_tail, _ = replay(args.source_root, harness, patch, 'native896', tokens=229768)
    assert native_tail['payload_failures']
    results.append(native_tail)
    first, state = replay(args.source_root, harness, patch, 'checkpoint')
    assert first['chunk_histogram'] == {1984: 64, 1600: 63, 1152: 1, 72: 1}
    for tokens, budget in ((229768, 384), (232000, 256)):
        growth, state = replay(args.source_root, harness, patch, 'checkpoint',
                              tokens=tokens, budget=budget, prior=state)
        assert growth['resumed_tokens'] > 0
        assert not growth['payload_failures']
        assert growth['non_kpool_aligned_chunk_ends'] == 0
        results.append(growth)
    assert results[0]['boundary_57344']['payload_tokens'] == 55552
    assert len(results[0]['payload_failures']) == 4
    print('PAYLOAD_RESULTS=' + json.dumps(results))
    print('PASS actual-source checkpoint payload identity, growth/resume, and kpool alignment; GPU untested')
