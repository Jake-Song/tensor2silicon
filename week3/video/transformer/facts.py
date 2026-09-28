"""Single source for displayed quantities; independently checked against model shapes."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from llm_roofline import spec


def calculate():
    cfg = spec.PRESETS['colab'].model
    D, F, L, V = cfg.d_model, cfg.d_ff, cfg.n_layers, cfg.vocab
    phases = [spec.PhaseConfig('prefill', 1, 2048, 2048), spec.PhaseConfig('decode', 1, 1, 2048)]
    data = dict(D=D, F=F, L=L, V=V, h=cfg.n_heads, d=cfg.head_dim,
                attention_weights=4*D*D, ffn_weights=3*D*F, norm_params_layer=4*D,
                layer_params=4*D*D+3*D*F+4*D, matmul_count=9*L+1,
                embedding_params=V*D, head_params=V*D)
    data['total_params'] = L*data['layer_params']+2*V*D+2*D
    for ph in phases:
        ops = spec.op_specs(cfg, ph)
        data[ph.name+'_matmul'] = sum(o.flops*o.count for o in ops if o.kind == 'matmul')
        data[ph.name+'_other'] = sum(o.flops*o.count for o in ops if o.kind != 'matmul')
        data[ph.name+'_ops'] = [dict(name=o.name, flops=o.flops, count=o.count, kind=o.kind,
                                   shapes=[list(a.shape) for a in o.args], mm_index=o.mm_index)
                                  for o in ops]
    return data


FACTS = calculate()


def verify():
    c = spec.PRESETS['colab'].model
    shapes = []
    spec.init_params(c, normal=lambda shape, mean, std: shapes.append(shape))
    assert sum(math.prod(s) for s in shapes) == FACTS['total_params'] == 542183424
    assert FACTS['layer_params'] == 51388416
    assert FACTS['matmul_count'] == 73
    assert FACTS['prefill_matmul'] == 2226940542976
    assert FACTS['decode_matmul'] == 1087373312
    for name, T in [('prefill', 2048), ('decode', 1)]:
        rows = FACTS[name+'_ops']
        assert [r['mm_index'] for r in rows if r['mm_index']] == list(range(1, 10))
        assert sum(r['count'] for r in rows if r['kind']=='matmul') == 73
        analytic = c.n_layers*(8*T*c.d_model**2+4*T*2048*c.d_model+6*T*c.d_model*c.d_ff)+2*T*c.d_model*c.vocab
        assert analytic == FACTS[name+'_matmul']
        for row in rows:
            if row['kind']=='matmul' and row['name'] not in ('qk','pv'):
                a,b = row['shapes']
                assert a[-1]==b[0]
                assert 2*math.prod(a)*b[1]==row['flops']
    short={o.name:o for o in spec.op_specs(c,spec.PhaseConfig('prefill',1,16,16))}
    long={o.name:o for o in spec.op_specs(c,spec.PhaseConfig('prefill',1,32,32))}
    for name in ('qk','pv'):
        assert long[name].flops==4*short[name].flops
    for name in ('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','lm_head'):
        assert long[name].flops==2*short[name].flops
        assert long[name].args[1].shape==short[name].args[1].shape
    return dict(params=FACTS['total_params'], matmuls=73, arithmetic='passed',
                convention='dense forward, 2 FLOPs/MAC; other op estimates from spec.py')


if __name__=='__main__':
    print(verify())
