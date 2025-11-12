from parser.pcfgs.pcfgs import PCFG_base
from parser.pcfgs.pcfg import PCFG


def _factorized_rule_to_full_logprob(rules):
    """Expand the factorized left/right parameterization into full binary rule logits."""
    left_child = torch.cat([rules['left_m'], rules['left_p']], dim=1)  # (B, NT+T, NT)
    right_child = torch.cat([rules['right_m'], rules['right_p']], dim=1)  # (B, NT+T, NT)

    # Reorder so the parent dimension is the second axis to match PCFG expectations.
    left_child = left_child.permute(0, 2, 1)   # (B, NT, NT+T)
    right_child = right_child.permute(0, 2, 1)  # (B, NT, NT+T)

    rule_prob = left_child.unsqueeze(-1) * right_child.unsqueeze(-2)  # (B, NT, NT+T, NT+T)
    rule_log = (rule_prob + 1e-9).log()
    return rule_log
from parser.pcfgs.fn import stripe, diagonal_copy_, checkpoint, diagonal, stripe_add_
import torch
from parser.triton.fn import _merge, _log_then_diagonal_copy_

class SimplePCFG_Triton(PCFG_base):
    def __init__(self):
        super(SimplePCFG_Triton, self).__init__()
        self._autograd_pcfg = PCFG()

    def _label_marginals_via_autograd(self, rules, lens):
        rule_log = _factorized_rule_to_full_logprob(rules)
        pcfg_rules = {
            'unary': rules['unary'],
            'rule': rule_log,
            'root': rules['root']
        }
        return self._autograd_pcfg._inside(pcfg_rules, lens, label_marginal=True)

    def loss(self, rules, lens):
        return self._inside(rules, lens)

    def label_marginals(self, rules, lens):
        return self._inside(rules, lens, label_marginal=True)

    @torch.enable_grad()
    def _inside(self, rules, lens, mbr=False, viterbi=False, marginal=False, s_span=None, entropy=False, label_marginal=False):
        if label_marginal and s_span is None:
            return self._label_marginals_via_autograd(rules, lens)
        assert viterbi is not True
        # B, L, r_p
        unary = rules['unary'].clone()
        # B, L, r_m
        root = rules['root'].exp()

        # r_m, r_m
        L = rules['left_m']
        R = rules['right_m']
        # r_p, r_p
        L_p = rules['left_p']
        R_p = rules['right_p']
        LR = torch.cat([L, R], dim=-1)
        # breakpoint()
        r_p = unary.shape[-1]
        r_m = L.shape[-2]

        batch, N, *_ = unary.shape
        N += 1
        # for estimating marginals.
        if s_span is None:
            indicator_dim = r_m if label_marginal else 1
            span_indicator = unary.new_zeros(batch, N, N, indicator_dim).requires_grad_(mbr or label_marginal)
        else:
            span_indicator = s_span
            if mbr or viterbi:
                span_indicator = span_indicator.detach().clone().requires_grad_(True)
            if span_indicator.dim() == 3:
                span_indicator = span_indicator.unsqueeze(-1)
            unary += diagonal(span_indicator, w=1).unsqueeze(-1)

        # normalizer = unary.new_zeros(batch, N, N).fill_(-1e9)

        with torch.no_grad():
            unary_max = unary.max(-1)[0]

        unary = (unary - unary_max.unsqueeze(-1)).exp()
        unary = torch.einsum('bnp, pq -> bnq',  unary ,torch.cat([L_p, R_p], dim=-1))

        alpha_c = unary.new_zeros(batch, N, N,  2, r_m)
        alpha_c = _log_then_diagonal_copy_(unary, unary_max, alpha_c)

        if label_marginal:
            diagonal(alpha_c[..., 0, :], 1).add_(diagonal(span_indicator, 1))

        # w: span width
        for w in range(2, N):
            n = N - w
            normalizer = alpha_c.new_zeros(batch, n)
            indicator = diagonal(span_indicator, w)
            out, normalizer = _merge(normalizer, indicator, alpha_c)
            if w < N-1:
                out = torch.einsum('blr, rq -> blq', out, LR)
                alpha_c = _log_then_diagonal_copy_(out, normalizer, alpha_c)

        logZ = (torch.einsum('bnr, br -> b', out, root) + 1e-9).log() + normalizer.squeeze(1)

        if not mbr and not viterbi and not label_marginal:
            return {'partition': logZ}

        if label_marginal:
            logZ.sum().backward()
            marginals = span_indicator.grad
            if marginals is not None and marginals.dim() == 4 and marginals.shape[-1] == 1:
                marginals = marginals.squeeze(-1)
            return {'partition': logZ, 'label_marginal': None if marginals is None else marginals.detach()}

        elif marginal:
            logZ.sum().backward()
            return {'marginal': span_indicator.grad}

        else:
            return {

                "prediction": self._get_prediction(logZ, span_indicator, lens, mbr=True),
                "partition": logZ
            }


class SimplePCFG_Triton_Batch(PCFG_base):
    def __init__(self):
        super(SimplePCFG_Triton_Batch, self).__init__()
        self._autograd_pcfg = PCFG()

    def _label_marginals_via_autograd(self, rules, lens):
        rule_log = _factorized_rule_to_full_logprob(rules)
        pcfg_rules = {
            'unary': rules['unary'],
            'rule': rule_log,
            'root': rules['root']
        }
        return self._autograd_pcfg._inside(pcfg_rules, lens, label_marginal=True)

    def loss(self, rules, lens):
        return self._inside(rules, lens)

    def label_marginals(self, rules, lens):
        return self._inside(rules, lens, label_marginal=True)

    @torch.enable_grad()
    def _inside(self, rules, lens, mbr=False, viterbi=False, marginal=False, s_span=None, entropy = False, label_marginal=False):
        if label_marginal and s_span is None:
            return self._label_marginals_via_autograd(rules, lens)
        assert viterbi is not True
        # B, L, r_p
        unary = rules['unary'].clone()
        # B, L, r_m
        root = rules['root'].exp()

        # r_m, r_m
        L = rules['left_m']
        R = rules['right_m']
        # r_p, r_p
        L_p = rules['left_p']
        R_p = rules['right_p']
        LR = torch.cat([L, R], dim=-1)
        r_p = unary.shape[-1]
        r_m = L.shape[-2]
        # breakpoint()
        batch, N, *_ = unary.shape
        N += 1
        # for estimating marginals.
        if s_span is None:
            indicator_dim = r_m if label_marginal else 1
            span_indicator = unary.new_zeros(batch, N, N, indicator_dim).requires_grad_(mbr or label_marginal)
        else:
            span_indicator = s_span
            if mbr or viterbi:
                span_indicator = span_indicator.detach().clone().requires_grad_(True)
            if span_indicator.dim() == 3:
                span_indicator = span_indicator.unsqueeze(-1)
            unary += diagonal(span_indicator, w=1).unsqueeze(-1)

        # normalizer = unary.new_zeros(batch, N, N).fill_(-1e9)

        with torch.no_grad():
            unary_max = unary.max(-1)[0]

        unary = (unary - unary_max.unsqueeze(-1)).exp()

        unary = torch.einsum('bnp, bpq -> bnq',  unary ,torch.cat([L_p, R_p], dim=-1))

        alpha_c = unary.new_zeros(batch, N, N,  2, r_m)

        alpha_c = _log_then_diagonal_copy_(unary, unary_max, alpha_c)

        if label_marginal:
            diagonal(alpha_c[..., 0, :], 1).add_(diagonal(span_indicator, 1))

        # w: span width
        for w in range(2, N):
            n = N - w
            normalizer = alpha_c.new_zeros(batch, n)

            indicator = diagonal(span_indicator, w)
            out, normalizer = _merge(normalizer, indicator, alpha_c)

            if w < N-1:
                out = torch.einsum('blr, brq -> blq', out, LR)
                alpha_c = _log_then_diagonal_copy_(out, normalizer, alpha_c)

        logZ = (torch.einsum('bnr, br -> b', out, root) + 1e-9).log() + normalizer.squeeze(1)

        if not mbr and not viterbi and not label_marginal:
            return {'partition': logZ}

        if label_marginal:
            logZ.sum().backward()
            marginals = span_indicator.grad
            if marginals is not None and marginals.dim() == 4 and marginals.shape[-1] == 1:
                marginals = marginals.squeeze(-1)
            return {'partition': logZ, 'label_marginal': None if marginals is None else marginals.detach()}


        elif marginal:

            logZ.sum().backward()

            return {'marginal': span_indicator.grad}

        else:
            return {

                "prediction": self._get_prediction(logZ, span_indicator, lens, mbr=True),
                "partition": logZ
            }
