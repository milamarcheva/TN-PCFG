from parser.pcfgs.pcfgs import PCFG_base, _normalize_label_marginals
from parser.pcfgs.fn import stripe, diagonal_copy_, diagonal, checkpoint
import torch


class PCFG(PCFG_base):

    def label_marginals(self, rules, lens):
        return self._inside(rules, lens, label_marginal=True)

    @torch.enable_grad()
    def _inside(self, rules, lens, viterbi=False, mbr=False, label_marginal=False):
        terms = rules['unary']
        rule = rules['rule']
        root = rules['root']

        batch, N, T = terms.shape
        N += 1
        NT = rule.shape[1]
        S = NT + T

        s = terms.new_zeros(batch, N, N, NT).fill_(-1e9)
        NTs = slice(0, NT)
        Ts = slice(NT, S)

        X_Y_Z = rule[:, :, NTs, NTs].reshape(batch, NT, NT * NT)
        X_y_Z = rule[:, :, Ts, NTs].reshape(batch, NT, NT * T)
        X_Y_z = rule[:, :, NTs, Ts].reshape(batch, NT, NT * T)
        X_y_z = rule[:, :, Ts, Ts].reshape(batch, NT, T * T)

        need_indicator = viterbi or mbr or label_marginal
        span_indicator = None
        if need_indicator:
            indicator_dim = NT if label_marginal else 1
            span_indicator = rule.new_zeros(batch, N, N, indicator_dim).requires_grad_(True)

        def contract(x, dim=-1):
            if viterbi:
                return x.max(dim)[0]
            else:
                return x.logsumexp(dim)

        # nonterminals: X Y Z
        # terminals: x y z
        # XYZ: X->YZ  XYz: X->Yz  ...
        @checkpoint
        def Xyz(y, z, rule):
            n = y.shape[1]
            b_n_yz = (y + z).reshape(batch, n, T * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule.unsqueeze(1))
            return b_n_x

        @checkpoint
        def XYZ(Y, Z, rule):
            n = Y.shape[1]
            b_n_yz = contract(Y[:, :, 1:-1, :].unsqueeze(-1) + Z[:, :, 1:-1, :].unsqueeze(-2), dim=2).reshape(batch, n, -1)
            b_n_x = contract(b_n_yz.unsqueeze(2) + rule.unsqueeze(1))
            return b_n_x

        @checkpoint
        def XYz(Y, z, rule):
            n = Y.shape[1]
            Y = Y[:, :, -1, :, None]
            b_n_yz = (Y + z).reshape(batch, n, NT * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule.unsqueeze(1))
            return b_n_x


        @checkpoint
        def XyZ(y, Z, rule):
            n = Z.shape[1]
            Z = Z[:, :, 0, None, :]
            b_n_yz = (y + Z).reshape(batch, n, NT * T)
            b_n_x = contract(b_n_yz.unsqueeze(-2) + rule.unsqueeze(1))
            return b_n_x


        for w in range(2, N):
            n = N - w

            Y_term = terms[:, :n, :, None]
            Z_term = terms[:, w - 1:, None, :]

            if w == 2:
                span_slice = None
                if span_indicator is not None:
                    span_slice = span_indicator[:, torch.arange(n), torch.arange(n) + w]
                base = Xyz(Y_term, Z_term, X_y_z)
                if span_slice is not None:
                    if span_slice.dim() == 3:
                        base = base + span_slice
                    else:
                        base = base + span_slice.unsqueeze(-1)
                diagonal_copy_(s, base, w)
                continue

            n = N - w
            x = terms.new_zeros(3, batch, n, NT).fill_(-1e9)

            Y = stripe(s, n, w - 1, (0, 1)).clone()
            Z = stripe(s, n, w - 1, (1, w), 0).clone()

            if w > 3:
                x[0].copy_(XYZ(Y, Z, X_Y_Z))

            x[1].copy_(XYz(Y, Z_term, X_Y_z))
            x[2].copy_(XyZ(Y_term, Z, X_y_Z))

            span_slice = None
            if span_indicator is not None:
                span_slice = span_indicator[:, torch.arange(n), torch.arange(n) + w]
            contracted = contract(x, dim=0)
            if span_slice is not None:
                if span_slice.dim() == 3:
                    contracted = contracted + span_slice
                else:
                    contracted = contracted + span_slice.unsqueeze(-1)
            diagonal_copy_(s, contracted, w)

        logZ = contract(s[torch.arange(batch), 0, lens] + root)

        if label_marginal:
            logZ.sum().backward()
            marginals = span_indicator.grad if span_indicator is not None else None
            if marginals is not None and marginals.shape[-1] == 1:
                marginals = marginals.squeeze(-1)
            marginals = _normalize_label_marginals(marginals)
            return {
                'partition': logZ,
                'label_marginal': None if marginals is None else marginals.detach()
            }

        if viterbi or mbr:
            prediction = self._get_prediction(logZ, span_indicator, lens, mbr=mbr)
            return {'partition': logZ,
                    'prediction': prediction}

        else:
            return {'partition': logZ}


class Faster_PCFG(PCFG_base):
    @torch.enable_grad()
    def _inside(self, rules, lens, viterbi=False, mbr=False):
        assert viterbi == False

        terms = rules['unary']
        rule = rules['rule']
        root = rules['root']


        batch, N, T = terms.shape
        N += 1
        NT = rule.shape[1]
        S = NT + T

        s = terms.new_zeros(batch, N, N, NT).fill_(-1e9)
        NTs = slice(0, NT)
        Ts = slice(NT, S)

        rule = rule.exp()
        X_Y_Z = rule[:, :, NTs, NTs].contiguous()
        X_y_Z = rule[:, :, Ts, NTs].contiguous()
        X_Y_z = rule[:, :, NTs, Ts].contiguous()
        X_y_z = rule[:, :, Ts, Ts].contiguous()

        span_indicator = rule.new_zeros(batch, N, N).requires_grad_(viterbi or mbr)

        def contract(x, dim=-1):
            if viterbi:
                return x.max(dim)[0]
            else:
                return x.logsumexp(dim)

        # nonterminals: X Y Z
        # terminals: x y z
        # XYZ: X->YZ
        @checkpoint
        def Xyz(y, z,  rule):
            y_normalizer = y.max(-1)[0]
            z_normalizer = z.max(-1)[0]
            y, z = (y-y_normalizer.unsqueeze(-1)).exp(), (z-z_normalizer.unsqueeze(-1)).exp()
            x = torch.einsum('bny, bnz, bxyz -> bnx', y, z, rule)
            x = ((x + 1e-9).log() + y_normalizer.unsqueeze(-1) + z_normalizer.unsqueeze(-1))
            return x

        @checkpoint
        def XYZ(Y, Z, rule):
            # n = Y.shape[1]
            Y = Y[:, :, 1:-1, :]
            Z = Z[:, :, 1:-1, :]
            Y_normalizer = Y.max(-1)[0]
            Z_normalizer = Z.max(-1)[0]
            Y, Z = (Y-Y_normalizer.unsqueeze(-1)).exp(), (Z-Z_normalizer.unsqueeze(-1)).exp()
            X = torch.einsum('bnwy, bnwz, bxyz -> bnwx', Y, Z, rule)
            X = (X + 1e-9).log() + Y_normalizer.unsqueeze(-1) + Z_normalizer.unsqueeze(-1)
            X = X.logsumexp(2)
            return X

        @checkpoint
        def XYz(Y, z, rule):
            Y = Y[:, :, -1, :]
            Y_normalizer = Y.max(-1)[0]
            z_normalizer = z.max(-1)[0]
            Y, z = (Y-Y_normalizer.unsqueeze(-1)).exp(), (z-z_normalizer.unsqueeze(-1)).exp()
            X = torch.einsum('bny, bnz, bxyz->bnx', Y, z, rule)
            X = (X + 1e-9).log() + Y_normalizer.unsqueeze(-1) + z_normalizer.unsqueeze(-1)
            return X

        @checkpoint
        def XyZ(y, Z, rule):
            Z = Z[:, :, 0, :]
            y_normalizer = y.max(-1)[0]
            Z_normalizer = Z.max(-1)[0]
            y, Z = (y-y_normalizer.unsqueeze(-1)).exp(), (Z-Z_normalizer.unsqueeze(-1)).exp()
            X = torch.einsum('bny, bnz, bxyz-> bnx', y, Z, rule)
            X = (X + 1e-9).log() + y_normalizer.unsqueeze(-1) + Z_normalizer.unsqueeze(-1)
            return X


        for w in range(2, N):
            n = N - w

            Y_term = terms[:, :n, :,]
            Z_term = terms[:, w - 1:, :]

            if w == 2:
                diagonal_copy_(s, Xyz(Y_term, Z_term, X_y_z) + span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1), w)
                continue

            n = N - w
            x = terms.new_zeros(3, batch, n, NT).fill_(-1e9)

            Y = stripe(s, n, w - 1, (0, 1)).clone()
            Z = stripe(s, n, w - 1, (1, w), 0).clone()

            if w > 3:
                x[0].copy_(XYZ(Y, Z, X_Y_Z))

            x[1].copy_(XYz(Y, Z_term, X_Y_z))
            x[2].copy_(XyZ(Y_term, Z, X_y_Z))

            diagonal_copy_(s, contract(x, dim=0) + span_indicator[:, torch.arange(n), torch.arange(n) + w].unsqueeze(-1), w)

        logZ = contract(s[torch.arange(batch), 0, lens] + root)

        if viterbi or mbr:
            prediction = self._get_prediction(logZ, span_indicator, lens, mbr=mbr)
            return {'partition': logZ,
                    'prediction': prediction}

        else:
            return {'partition': logZ}

