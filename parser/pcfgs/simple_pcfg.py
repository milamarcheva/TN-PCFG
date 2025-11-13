import torch

from parser.pcfgs.fn import diagonal
from parser.pcfgs.pcfgs import PCFG_base
from parser.triton.fn import _merge, _log_then_diagonal_copy_

class SimplePCFG_Triton(PCFG_base):
    def __init__(self):
        super(SimplePCFG_Triton, self).__init__()

    def loss(self, rules, lens):
        return self._inside(rules, lens)

    def label_marginals(self, rules, lens):
        return self._inside(rules, lens, label_marginal=True)

    @torch.enable_grad()
    def _inside(self, rules, lens, mbr=False, viterbi=False, marginal=False, s_span=None, entropy=False, label_marginal=False):
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

        if label_marginal:
            base_indicator = diagonal(span_indicator, 1)
            unary_log = unary.clamp_min(1e-9).log() + unary_max.unsqueeze(-1)
            unary_log = unary_log.view(batch, N - 1, 2, r_m)
            unary_log = unary_log + base_indicator.unsqueeze(2)
            unary_log = unary_log.view(batch, N - 1, 2 * r_m)
            unary_max = unary_log.max(-1)[0]
            unary = torch.exp(unary_log - unary_max.unsqueeze(-1))

        alpha_c = unary.new_zeros(batch, N, N,  2, r_m)
        alpha_c = _log_then_diagonal_copy_(unary, unary_max, alpha_c)

        # w: span width
        for w in range(2, N):
            n = N - w
            normalizer = alpha_c.new_zeros(batch, n)
            indicator = diagonal(span_indicator, w)
            out, normalizer = _merge(normalizer, indicator, alpha_c)

            parent_out = out
            parent_normalizer = normalizer

            if label_marginal:
                parent_log = parent_out.clamp_min(1e-9).log() + parent_normalizer.unsqueeze(-1)
                if indicator.dim() == 3:
                    parent_log = parent_log + indicator
                else:
                    parent_log = parent_log + indicator.unsqueeze(-1)
                parent_normalizer = parent_log.max(-1)[0]
                parent_out = torch.exp(parent_log - parent_normalizer.unsqueeze(-1))

            out = parent_out
            normalizer = parent_normalizer

            if w < N-1:
                oriented = torch.einsum('blr, rq -> blq', parent_out, LR)
                alpha_c = _log_then_diagonal_copy_(oriented, parent_normalizer, alpha_c)

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

    def loss(self, rules, lens):
        return self._inside(rules, lens)

    def label_marginals(self, rules, lens):
        return self._inside(rules, lens, label_marginal=True)

    @torch.enable_grad()
    def _inside(self, rules, lens, mbr=False, viterbi=False, marginal=False, s_span=None, entropy = False, label_marginal=False):
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

        if label_marginal:
            base_indicator = diagonal(span_indicator, 1)
            unary_log = unary.clamp_min(1e-9).log() + unary_max.unsqueeze(-1)
            unary_log = unary_log.view(batch, N - 1, 2, r_m)
            unary_log = unary_log + base_indicator.unsqueeze(2)
            unary_log = unary_log.view(batch, N - 1, 2 * r_m)
            unary_max = unary_log.max(-1)[0]
            unary = torch.exp(unary_log - unary_max.unsqueeze(-1))

        alpha_c = unary.new_zeros(batch, N, N,  2, r_m)

        alpha_c = _log_then_diagonal_copy_(unary, unary_max, alpha_c)

        # w: span width
        for w in range(2, N):
            n = N - w
            normalizer = alpha_c.new_zeros(batch, n)

            indicator = diagonal(span_indicator, w)
            out, normalizer = _merge(normalizer, indicator, alpha_c)

            parent_out = out
            parent_normalizer = normalizer

            if label_marginal:
                parent_log = parent_out.clamp_min(1e-9).log() + parent_normalizer.unsqueeze(-1)
                if indicator.dim() == 3:
                    parent_log = parent_log + indicator
                else:
                    parent_log = parent_log + indicator.unsqueeze(-1)
                parent_normalizer = parent_log.max(-1)[0]
                parent_out = torch.exp(parent_log - parent_normalizer.unsqueeze(-1))

            out = parent_out
            normalizer = parent_normalizer

            if w < N-1:
                oriented = torch.einsum('blr, brq -> blq', parent_out, LR)
                alpha_c = _log_then_diagonal_copy_(oriented, parent_normalizer, alpha_c)

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
