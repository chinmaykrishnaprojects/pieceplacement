"""adaptive_budget — spend LM calls only where the policy is unsure.

Idea
----
The chess-GPT policy is a *human-move* distribution. Its own shape tells us how
hard the position is: when one move carries most of the mass and cheap board
features show nothing tactical, a second LM call buys nothing. When the mass is
spread, or when a free material search disagrees with the policy, or when the
position is forcing, the extra calls pay for themselves.

So:
  0 calls  - forced move (one legal reply) or a free mate-in-1.
  1 call   - the common case. The single policy is re-ranked by a *free*
             material + PST + quiescence search: score = qval + K*log(p/p_top).
             This is a pure blunder filter; it can only overrule the policy's
             favourite when a cheap search says it drops material.
  1+k call - "sharp" positions only. Each of the top-k policy moves gets one
             child policy call, giving the human-likely replies; those replies
             plus the material-best replies are searched a further ply. This is
             a policy-pruned 4-ply line over 2-3 candidate moves, and it is
             human-legible: the player literally considered k named moves and
             their principal replies.

Everything except `oracle.policy` is free arithmetic, so the blunder filter and
the leaf evaluation cost no budget at all.
"""

import math

import chess

INF = 10 ** 9
MATE = 100000

PV = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Piece-square tables, written a8..h1 (rank 8 first), white's point of view.
_PST_P = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0]
_PST_N = [
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50]
_PST_B = [
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -20,-10,-10,-10,-10,-10,-10,-20]
_PST_R = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0]
_PST_Q = [
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
     0,  0,  5,  5,  5,  5,  0, -5,
   -10,  5,  5,  5,  5,  5,  0,-10,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20]
_PST_K = [
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20]
_PST_KE = [
   -50,-40,-30,-20,-20,-30,-40,-50,
   -30,-20,-10,  0,  0,-10,-20,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-30,  0,  0,  0,  0,-30,-30,
   -50,-30,-30,-30,-30,-30,-30,-50]

_PST = {
    chess.PAWN: _PST_P, chess.KNIGHT: _PST_N, chess.BISHOP: _PST_B,
    chess.ROOK: _PST_R, chess.QUEEN: _PST_Q,
}

_MIRROR = [chess.square_mirror(s) for s in range(64)]

# --- tuning -----------------------------------------------------------------
K_POL = 110.0      # centipawns charged per nat of policy log-prob lost
P_FLOOR = 0.035    # candidate must hold this fraction of the top move's mass
MAX_CANDS = 6
QMAX = 5           # quiescence ply cap
NODE_CAP_QUIET = 7000
NODE_CAP_SHARP = 22000
# sharpness thresholds
CONF_P = 0.60      # top-move probability above which the position looks obvious
CONF_MARGIN = 0.33  # p1 - p2 above which the policy is decisive
DISAGREE_CP = 45   # static-vs-policy disagreement that makes a position sharp
TACTIC_CP = 140    # a capture this favourable (by raw values) means tactics


def _pgn_push(pgn, board, move):
    """Extend a Karvonen-format PGN prefix exactly the way the arena does."""
    san = board.san(move)
    if board.turn == chess.WHITE:
        return pgn + "%d.%s " % (board.fullmove_number, san)
    return pgn + san + " "


class Player:
    def __init__(self, oracle, budget):
        self.oracle = oracle
        self.budget = max(1, int(budget))
        self.nodes = 0
        self.cap = NODE_CAP_QUIET
        self.last_lines = []      # human-readable trace of what was considered

    # ------------------------------------------------------------------ eval
    def _eval(self, board):
        """Static score in centipawns from the side-to-move's point of view."""
        wm = 0
        bm = 0
        wp = 0
        bp = 0
        wb = 0
        bb = 0
        pm = board.piece_map()
        # endgame-ness from non-pawn material
        heavy = 0
        for sq, pc in pm.items():
            t = pc.piece_type
            if t != chess.PAWN and t != chess.KING:
                heavy += PV[t]
        endgame = heavy <= 1800
        for sq, pc in pm.items():
            t = pc.piece_type
            if pc.color == chess.WHITE:
                i = _MIRROR[sq]
                wm += PV[t]
                if t == chess.KING:
                    wp += (_PST_KE if endgame else _PST_K)[i]
                else:
                    wp += _PST[t][i]
                    if t == chess.BISHOP:
                        wb += 1
            else:
                bm += PV[t]
                if t == chess.KING:
                    bp += (_PST_KE if endgame else _PST_K)[sq]
                else:
                    bp += _PST[t][sq]
                    if t == chess.BISHOP:
                        bb += 1
        if wb >= 2:
            wp += 30
        if bb >= 2:
            bp += 30
        s = (wm + wp) - (bm + bp)
        return s if board.turn == chess.WHITE else -s

    @staticmethod
    def _order(board, mv):
        s = 0
        if board.is_capture(mv):
            victim = board.piece_type_at(mv.to_square)
            v = PV[victim] if victim else 100     # en-passant
            att = board.piece_type_at(mv.from_square)
            s += 1000 + v - (PV[att] if att else 0) // 10
        if mv.promotion:
            s += 800 + PV[mv.promotion]
        return s

    def _qsearch(self, board, alpha, beta, ply):
        self.nodes += 1
        if self.nodes > self.cap:
            return self._eval(board)
        in_check = board.is_check()
        if in_check:
            moves = list(board.legal_moves)
            if not moves:
                return -MATE + ply
            if ply >= QMAX:
                return self._eval(board)
        else:
            stand = self._eval(board)
            if ply >= QMAX:
                return stand
            if stand >= beta:
                return stand
            if stand > alpha:
                alpha = stand
            moves = [m for m in board.legal_moves
                     if board.is_capture(m) or m.promotion]
            if not moves:
                return stand
            # delta pruning
            keep = []
            for m in moves:
                victim = board.piece_type_at(m.to_square)
                gain = (PV[victim] if victim else 100)
                if m.promotion:
                    gain += PV[m.promotion]
                if stand + gain + 200 < alpha:
                    continue
                keep.append(m)
            if not keep:
                return stand
            moves = keep
        moves.sort(key=lambda m: self._order(board, m), reverse=True)
        best = -INF if in_check else alpha
        for m in moves:
            board.push(m)
            v = -self._qsearch(board, -beta, -alpha, ply + 1)
            board.pop()
            if v > best:
                best = v
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
            if self.nodes > self.cap:
                break
        return best

    def _negamax(self, board, depth, alpha, beta, ply=0):
        if self.nodes > self.cap:
            return self._eval(board)
        if depth <= 0:
            return self._qsearch(board, alpha, beta, ply)
        self.nodes += 1
        moves = list(board.legal_moves)
        if not moves:
            return (-MATE + ply) if board.is_check() else 0
        moves.sort(key=lambda m: self._order(board, m), reverse=True)
        best = -INF
        for m in moves:
            board.push(m)
            v = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            if v > best:
                best = v
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
            if self.nodes > self.cap:
                break
        return best

    # ------------------------------------------------------- root evaluation
    def _value_after(self, board, mv, depth):
        """Value of `mv` for the mover, via depth-`depth` search + quiescence."""
        board.push(mv)
        try:
            if board.is_checkmate():
                return MATE
            if (board.is_stalemate() or board.is_insufficient_material()
                    or board.is_repetition(3) or board.halfmove_clock >= 100):
                return 0
            return -self._negamax(board, depth, -INF, INF, 1)
        finally:
            board.pop()

    def _tactical(self, board):
        """Cheap flag: is there real material tension on the board?"""
        if board.is_check():
            return True
        best = 0
        for m in board.legal_moves:
            if not board.is_capture(m):
                continue
            victim = board.piece_type_at(m.to_square)
            v = PV[victim] if victim else 100
            att = board.piece_type_at(m.from_square)
            gain = v - (PV[att] if att else 0)
            if gain > best:
                best = gain
        if best >= TACTIC_CP:
            return True
        # also: is one of OUR pieces attacked by a cheaper enemy piece?
        us = board.turn
        for sq, pc in board.piece_map().items():
            if pc.color != us or pc.piece_type == chess.KING:
                continue
            atk = board.attackers(not us, sq)
            if not atk:
                continue
            if not board.attackers(us, sq):
                if PV[pc.piece_type] >= 300:
                    return True
            else:
                cheapest = min(PV[board.piece_type_at(a)] for a in atk)
                if PV[pc.piece_type] - cheapest >= TACTIC_CP:
                    return True
        return False

    # ------------------------------------------------------------------ play
    def play(self, board, pgn):
        legal = list(board.legal_moves)
        if not legal:
            return None
        if len(legal) == 1:
            self.last_lines = [("forced", legal[0], 0.0)]
            return legal[0]

        # Free win: a mate in one never needs an LM call.
        for m in legal:
            board.push(m)
            mate = board.is_checkmate()
            board.pop()
            if mate:
                self.last_lines = [("mate-in-1", m, MATE)]
                return m

        self.nodes = 0
        self.cap = NODE_CAP_QUIET

        try:
            pol = self.oracle.policy(board, pgn)
        except Exception:       # BudgetExhausted or a model hiccup
            pol = None
        used = 1

        items = []
        if pol:
            for m, p in pol.items():
                if m in board.legal_moves and p > 0:
                    items.append((m, float(p)))
        if not items:
            # No usable policy: fall back to our own free search.
            self.cap = NODE_CAP_SHARP
            best, bv = legal[0], -INF
            for m in legal:
                v = self._value_after(board, m, 1)
                if v > bv:
                    bv, best = v, m
            self.last_lines = [("no-policy", best, bv)]
            return best

        items.sort(key=lambda x: -x[1])
        p_top = items[0][1]
        policy_best = items[0][0]
        p2 = items[1][1] if len(items) > 1 else 0.0
        cands = [(m, p) for (m, p) in items if p >= P_FLOOR * p_top][:MAX_CANDS]
        # The free search costs nothing, so always look at >=3 alternatives:
        # a confident policy is exactly when an unchecked blunder is expensive.
        if len(cands) < 3:
            cands = items[:min(3, len(items))]
        if len(cands) == 1:
            self.last_lines = [("only-move", cands[0][0], 0.0)]
            return cands[0][0]

        # ---- stage 1: free blunder filter (no LM cost) --------------------
        scored = []
        for m, p in cands:
            v = self._value_after(board, m, 1)
            bonus = K_POL * math.log(p / p_top)
            scored.append([m, p, v, v + bonus])
        scored.sort(key=lambda r: -r[3])
        static_best = scored[0][0]
        best_move = static_best

        # ---- stage 2: is this position worth more LM calls? ---------------
        remaining = self.budget - used
        if remaining <= 0:
            self.last_lines = [(board.san(r[0]), r[0], r[3]) for r in scored[:3]]
            return best_move

        v_pol = next(r[2] for r in scored if r[0] == policy_best)
        v_best = max(r[2] for r in scored)
        confident = (p_top >= CONF_P) or ((p_top - p2) >= CONF_MARGIN)
        disagree = (static_best != policy_best) and (v_best - v_pol >= DISAGREE_CP)
        spread = scored[0][3] - scored[min(1, len(scored) - 1)][3]

        sharp = (not confident) or disagree or self._tactical(board) \
            or spread < 25.0
        if not sharp:
            self.last_lines = [(board.san(r[0]), r[0], r[3]) for r in scored[:3]]
            return best_move

        # ---- stage 3: policy-pruned lookahead over the top-k moves --------
        self.cap = NODE_CAP_SHARP
        k = min(remaining, 3, len(scored))
        refined = []
        try:
            for row in scored[:k]:
                m, p = row[0], row[1]
                try:
                    child_pgn = _pgn_push(pgn, board, m)
                except Exception:
                    child_pgn = None
                board.push(m)
                try:
                    if board.is_checkmate():
                        refined.append([m, p, MATE, MATE])
                        continue
                    if (board.is_stalemate() or board.is_insufficient_material()
                            or board.is_repetition(3)):
                        refined.append([m, p, 0, 0 + K_POL * math.log(p / p_top)])
                        continue
                    sub = None
                    if child_pgn is not None:
                        sub = self.oracle.policy(board, child_pgn)
                    replies = []
                    if sub:
                        for rm, rp in sorted(sub.items(), key=lambda x: -x[1]):
                            if rm in board.legal_moves:
                                replies.append(rm)
                            if len(replies) >= 3:
                                break
                    # add the materially most testing replies
                    mat = sorted(board.legal_moves,
                                 key=lambda x: self._order(board, x),
                                 reverse=True)[:3]
                    for rm in mat:
                        if rm not in replies:
                            replies.append(rm)
                    worst = INF
                    for rm in replies:
                        # value for US after opponent plays rm, one more ply
                        v = -self._value_after(board, rm, 1)
                        if v < worst:
                            worst = v
                    if worst == INF:
                        worst = row[2]
                    refined.append([m, p, worst,
                                    worst + K_POL * math.log(p / p_top)])
                finally:
                    board.pop()
        except Exception:
            # Budget ran out mid-way: keep whatever we refined, fall back for
            # the rest to the stage-1 scores.
            pass

        done = {r[0] for r in refined}
        merged = refined + [r for r in scored if r[0] not in done]
        merged.sort(key=lambda r: -r[3])
        self.last_lines = [(board.san(r[0]), r[0], r[3]) for r in merged[:3]]
        return merged[0][0]


def create_player(oracle, budget):
    return Player(oracle, budget)
