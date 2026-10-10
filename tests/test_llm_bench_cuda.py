"""CUDA C++ correctness, mutation, compile and graph regression tests."""
import importlib.util
import os
import unittest

RUN = os.environ.get('RUN_LLM_BENCH_GPU_TESTS') == '1'


@unittest.skipUnless(RUN and importlib.util.find_spec('torch'), 'opt-in CUDA GPU tests')
class CudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise unittest.SkipTest('BF16 CUDA required')
        from llm_bench import cuda_ops
        cls.torch, cls.ops = torch, cuda_ops
        cuda_ops.extension()

    def test_small_ops_and_registration(self):
        t, o = self.torch, self.ops
        x = t.randn(2, 3, 96, device='cuda', dtype=t.bfloat16)
        w, b = t.randn(96, device='cuda', dtype=t.bfloat16), t.randn(96, device='cuda', dtype=t.bfloat16)
        for fn, args in ((o.layernorm, (x, w, b)), (o.add_layernorm, (x, x, w, b)),
                         (o.swiglu, (x,)), (o.select_token, (x,))):
            t.library.opcheck(fn, args)
        from llm_bench.bench import validate_tensor
        validate_tensor(o.layernorm(x,w,b), t.nn.functional.layer_norm(x.float(),(96,),w.float(),b.float()), 'LN')
        summed, norm = o.add_layernorm(x,x,w,b)
        self.assertTrue(t.equal(summed, x+x))
        validate_tensor(norm, t.nn.functional.layer_norm(summed.float(),(96,),w.float(),b.float()), 'residual LN')
        validate_tensor(o.swiglu(x), t.nn.functional.silu(x[...,:48].float())*x[...,48:].float(), 'SwiGLU')
        equal = t.zeros(2,3,513,device='cuda',dtype=t.bfloat16)
        equal[:,-1,257] = equal[:,-1,511] = 3
        self.assertTrue(t.equal(o.select_token(equal), t.full((2,1),257,device='cuda')))
        for m,k,n in ((1,96,129),(8,65,73),(33,96,127)):
            a=t.randn(m,k,device='cuda',dtype=t.bfloat16);w=t.randn(k,n,device='cuda',dtype=t.bfloat16)
            validate_tensor(o.linear(a,w),a.float()@w.float(),'GEMM')

    def test_mutating_operator_contracts(self):
        t, o = self.torch, self.ops
        token = t.tensor([[3], [7]], device='cuda', dtype=t.int64)
        destination = t.zeros_like(token)
        position = t.tensor(0, device='cuda', dtype=t.int64)
        history = t.zeros(2, 4, device='cuda', dtype=t.int64)
        logits = t.randn(2, 3, 129, device='cuda', dtype=t.bfloat16)
        src = t.randn(2, 2, 3, 32, device='cuda', dtype=t.bfloat16)
        dst = t.zeros(2, 2, 7, 32, device='cuda', dtype=t.bfloat16)
        for fn, args in ((o.copy_token, (token, destination)), (o.advance, (position,)),
                         (o.copy_prefix, (src, dst)), (o.record_token, (token, history, 2)),
                         (o.select_token_into, (logits, destination))):
            t.library.opcheck(fn, args)
        o.record_token(token, history, 1)
        self.assertTrue(t.equal(history[:, 1:2], token))
        table = t.randn(31, 32, device='cuda', dtype=t.bfloat16)
        ids = t.randint(31, (2, 8), device='cuda')[:, ::2]
        t.library.opcheck(o.embedding, (ids, table))
        self.assertTrue(t.equal(o.embedding(ids, table), table[ids]))
        view = logits[:, :, ::2]
        self.assertTrue(t.equal(o.select_token(view), view[:, -1].argmax(-1, keepdim=True)))

    def test_attention_tails_gqa_and_future_mask(self):
        t,o=self.torch,self.ops
        from llm_bench.bench import validate_tensor
        for heads,kv,h in ((4,4,32),(4,2,64),(2,1,128)):
            q=t.randn(2,heads,33,h,device='cuda',dtype=t.bfloat16)
            k=t.randn(2,kv,33,h,device='cuda',dtype=t.bfloat16);v=t.randn_like(k)
            ref=t.nn.functional.scaled_dot_product_attention(q.float(),k.float(),v.float(),is_causal=True,enable_gqa=heads!=kv).transpose(1,2).reshape(2,33,heads*h)
            validate_tensor(o.attention(q,k,v),ref,'prefill attention')
            pos=t.tensor(16,device='cuda',dtype=t.int64);qd=q[:,:,:1].contiguous()
            ref=t.nn.functional.scaled_dot_product_attention(qd.float(),k[:,:,:17].float(),v[:,:,:17].float(),enable_gqa=heads!=kv).transpose(1,2).reshape(2,1,heads*h)
            actual=o.attention(qd,k,v,pos)
            validate_tensor(actual,ref,'decode attention')
            k[:,:,17:]=100;v[:,:,17:]=-100
            self.assertTrue(t.equal(actual,o.attention(qd,k,v,pos)))

    def test_full_model_compile_mutation_and_graph(self):
        t=self.torch
        from llm_bench.bench import make_runner, make_cache, map_tree, reference, validate_output, Captured
        from llm_roofline import spec, torch_llm
        cfg=spec.PRESETS['tiny'].model
        with t.inference_mode():
            params=torch_llm.random_params(cfg,t.bfloat16,t.device('cuda'),seed=42)
            fp32=map_tree(params,lambda x:x.float())
            tokens=t.randint(cfg.vocab,(2,7),device='cuda')
            for name in ('cuda','hybrid-cuda'):
                runner=make_runner(name,cfg,params,12)
                expected=reference(cfg,fp32,tokens)
                actual=runner.prefill(tokens)
                validate_output(actual,expected)
                g=t.Generator(device='cuda').manual_seed(9)
                cache=make_cache(cfg,2,12,g,random=False)
                runner.copy_prefill_cache(actual[1],cache)
                prefix=map_tree(cache,lambda x:x.clone())
                token=tokens[:,:1].contiguous();pos=t.tensor(7,device='cuda',dtype=t.int64)
                # Position is a GPU tensor shared by every replay.
                graph=Captured(lambda: runner.decode(token,cache,pos))
                for position in (7,8):
                    pos.fill_(position)
                    token.copy_(t.randint(cfg.vocab,token.shape,device='cuda'))
                    expected=reference(cfg,fp32,token,cache,position)
                    validate_output(graph(),expected,position+1)
                for pair,old in zip(cache,prefix):
                    for a,b in zip(pair,old):
                        self.assertTrue(t.equal(a[:,:,:7],b[:,:,:7]))
                        self.assertTrue(t.equal(a[:,:,9:],b[:,:,9:]))
                pos.fill_(7)
                advance_graph=Captured(lambda:runner.advance(pos))
                pos.fill_(7);advance_graph();advance_graph()
                self.assertEqual(pos.item(),9)


if __name__ == '__main__':
    unittest.main()
