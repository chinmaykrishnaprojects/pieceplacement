"""blunder_filter: one LM call + a pure-arithmetic tactical safety filter.

Thesis: the human-trained policy already picks reasonable *strategic* moves, but
with no search it hangs pieces and misses one-movers. Everything below costs
ZERO budget: static exchange evaluation, a capture-only quiescence, a mate-in-1
scan for both sides. We take ONE policy call, keep its top few moves, and play
the highest-probability move that survives the tactical filter -- unless a move
outside that set wins clear material or mates.

Legibility: self.last_report holds the (move, policy prob, tactical value in cp)
triple for every candidate actually considered, so a human can read off exactly
why the LM's favourite was overruled.
"""

import chess

PV = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
      chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000}
SEE_PV = dict(PV)
SEE_PV[chess.KING] = 10000

LVA_ORDER = (chess.PAWN, chess.KNIGHT, chess.BISHOP,
             chess.ROOK, chess.QUEEN, chess.KING)
PIECE_TYPES = (chess.PAWN, chess.KNIGHT, chess.BISHOP,
               chess.ROOK, chess.QUEEN)

MATE = 30000
INF = 1 << 20

# how many centipawns one unit of policy probability is worth
POLICY_WEIGHT = 150.0
# tactical drops smaller than this are treated as evaluation noise and the
# policy is trusted; beyond it, every lost centipawn counts double, so no
# amount of policy confidence buys a genuinely hung piece
NOISE_BAND = 60.0
LOSS_SCALE = 2.0
# max candidates pulled from the policy distribution
MAX_CANDIDATES = 6
# quiescence depth (in plies) and a hard node cap per move
QDEPTH = 4
NODE_CAP = 3000
# a capture outside the policy's shortlist must win at least this much
FREE_MATERIAL = 95


class Player:
    def __init__(self, oracle, budget):
        self.oracle = oracle
        self.budget = budget
        self.nodes = 0
        self.last_report = []

    # ---------------- static helpers (all free, no budget) ----------------

    @staticmethod
    def _material(board):
        """Material balance in centipawns, from the side-to-move's view."""
        total = 0
        for pt in PIECE_TYPES:
            v = PV[pt]
            total += v * chess.popcount(board.pieces_mask(pt, chess.WHITE))
            total -= v * chess.popcount(board.pieces_mask(pt, chess.BLACK))
        return total if board.turn == chess.WHITE else -total

    def _see(self, board, move):
        """Static exchange evaluation: cp won by initiating this capture."""
        to_sq = move.to_square
        occ = board.occupied & ~chess.BB_SQUARES[move.from_square]
        if board.is_en_passant(move):
            gain0 = PV[chess.PAWN]
            cap_sq = to_sq - 8 if board.turn == chess.WHITE else to_sq + 8
            occ &= ~chess.BB_SQUARES[cap_sq]
        else:
            victim = board.piece_type_at(to_sq)
            gain0 = SEE_PV[victim] if victim else 0
        occ |= chess.BB_SQUARES[to_sq]

        mover = board.piece_type_at(move.from_square)
        if mover is None:
            return 0
        if move.promotion:
            gain0 += PV[move.promotion] - PV[chess.PAWN]
            att_val = SEE_PV[move.promotion]
        else:
            att_val = SEE_PV[mover]

        gains = [gain0]
        side = not board.turn
        d = 0
        while True:
            atts = board.attackers_mask(side, to_sq, occ) & occ
            if not atts:
                break
            sq = None
            cpt = None
            for pt in LVA_ORDER:
                sub = atts & board.pieces_mask(pt, side)
                if sub:
                    sq = chess.lsb(sub)
                    cpt = pt
                    break
            if sq is None:
                break
            if cpt == chess.KING:
                rest = occ & ~chess.BB_SQUARES[sq]
                if board.attackers_mask(not side, to_sq, rest) & rest:
                    break
            d += 1
            gains.append(att_val - gains[d - 1])
            att_val = SEE_PV[cpt]
            occ &= ~chess.BB_SQUARES[sq]
            side = not side
        while d > 0:
            gains[d - 1] = -max(-gains[d - 1], gains[d])
            d -= 1
        return gains[0]

    @staticmethod
    def _mate_in_one(board):
        """A move for the side to move that mates immediately, or None."""
        for mv in board.legal_moves:
            if not board.gives_check(mv):
                continue
            board.push(mv)
            done = board.is_checkmate()
            board.pop()
            if done:
                return mv
        return None

    # ---------------- capture-only quiescence (free) ----------------

    def _tactical(self, board, alpha, beta, depth, ply):
        """Negamax over captures/promotions, from the side-to-move's view."""
        self.nodes += 1
        if self.nodes > NODE_CAP:
            return self._material(board)

        in_check = board.is_check()
        if in_check and depth > -2:
            best = -INF
            for mv in board.legal_moves:
                board.push(mv)
                sc = -self._tactical(board, -beta, -alpha, depth - 1, ply + 1)
                board.pop()
                if sc > best:
                    best = sc
                    if best > alpha:
                        alpha = best
                        if alpha >= beta:
                            break
            if best == -INF:
                return -MATE + ply
            return best

        stand = self._material(board)
        if depth <= 0 or in_check:
            return stand
        if stand >= beta:
            return stand
        if stand > alpha:
            alpha = stand
        best = stand

        moves = []
        for mv in board.generate_legal_captures():
            victim = board.piece_type_at(mv.to_square)
            vv = PV[victim] if victim else PV[chess.PAWN]
            att = board.piece_type_at(mv.from_square)
            moves.append((vv * 16 - PV.get(att, 0) // 10, mv))
        for mv in board.generate_legal_moves(board.pawns, ~board.occupied):
            if mv.promotion == chess.QUEEN:
                moves.append((PV[chess.QUEEN] * 16, mv))
        moves.sort(key=lambda x: -x[0])

        for _, mv in moves:
            if self._see(board, mv) < 0:
                continue
            board.push(mv)
            sc = -self._tactical(board, -beta, -alpha, depth - 1, ply + 1)
            board.pop()
            if sc > best:
                best = sc
                if best > alpha:
                    alpha = best
                    if alpha >= beta:
                        break
            if self.nodes > NODE_CAP:
                break
        return best

    def _value_of(self, board, move):
        """Tactical value of `move` for the mover, in centipawns."""
        board.push(move)
        try:
            if board.is_checkmate():
                return MATE
            if board.is_stalemate() or board.is_insufficient_material():
                edge = -self._material(board)
                return -300 if edge > 150 else 0
            if self._mate_in_one(board) is not None:
                return -MATE + 2
            return -self._tactical(board, -INF, INF, QDEPTH, 1)
        finally:
            board.pop()

    @staticmethod
    def _score(row, vmax):
        """Combine tactical value with policy probability.

        A drop of `dv` centipawns relative to the best candidate is forgiven up
        to NOISE_BAND (the quiescence is not a real evaluation, so small gaps
        mean nothing), and charged double beyond it.
        """
        dv = row[2] - vmax
        if dv < -NOISE_BAND:
            dv = -NOISE_BAND + (dv + NOISE_BAND) * LOSS_SCALE
        return dv + POLICY_WEIGHT * row[1]

    def _rank(self, results):
        vmax = max(r[2] for r in results)
        results.sort(key=lambda r: -self._score(r, vmax))

    # ---------------- the player ----------------

    def play(self, board, pgn):
        legal = list(board.legal_moves)
        if not legal:
            return None
        if len(legal) == 1:
            return legal[0]

        # 0. free mate in one -- costs no budget at all.
        mate = self._mate_in_one(board)
        if mate is not None:
            self.last_report = [(mate, 1.0, MATE)]
            return mate

        # 1. the single policy call.
        pol = {}
        try:
            raw = self.oracle.policy(board, pgn)
            if raw:
                legal_set = set(legal)
                pol = {m: float(p) for m, p in raw.items() if m in legal_set}
        except Exception:
            pol = {}

        ranked = sorted(pol.items(), key=lambda kv: -kv[1])[:MAX_CANDIDATES]
        cands = {m: p for m, p in ranked}
        if not cands:
            # no policy: fall back to a purely tactical shortlist
            scored = []
            for mv in legal:
                s = self._see(board, mv) if board.is_capture(mv) else 0
                scored.append((s, mv))
            scored.sort(key=lambda x: -x[0])
            cands = {mv: 0.0 for _, mv in scored[:MAX_CANDIDATES]}

        # 2. add clearly-winning captures the policy may have overlooked.
        extras = []
        for mv in legal:
            if mv in cands:
                continue
            if board.is_capture(mv) or mv.promotion == chess.QUEEN:
                s = self._see(board, mv)
                if s >= FREE_MATERIAL:
                    extras.append((s, mv))
        extras.sort(key=lambda x: -x[0])
        for _, mv in extras[:3]:
            cands[mv] = 0.0

        # 3. score every candidate with free tactics.
        self.nodes = 0
        results = []
        for mv, p in cands.items():
            self.nodes = 0
            v = self._value_of(board, mv)
            results.append([mv, p, v])

        # 4. optional refinement when budget allows: ask the policy what the
        #    opponent would actually reply to our leading candidates.
        if self.budget > 1 and len(results) > 1:
            self._rank(results)
            top = self._score(results[0], max(r[2] for r in results))
            vmax = max(r[2] for r in results)
            for row in results[:3]:
                if top - self._score(row, vmax) > 130:
                    break
                if abs(row[2]) > MATE - 100:
                    continue
                refined = self._probe_reply(board, pgn, row[0])
                if refined is None:
                    break
                row[2] = (row[2] + refined) // 2

        self._rank(results)
        self.last_report = [(m, p, v) for m, p, v in results]
        best = results[0][0]
        return best if best in board.legal_moves else legal[0]

    def _probe_reply(self, board, pgn, move):
        """One extra LM call: score `move` against the opponent's likely replies."""
        try:
            san = board.san(move)
        except Exception:
            return None
        if board.turn == chess.WHITE:
            child_pgn = pgn + "%d.%s " % (board.fullmove_number, san)
        else:
            child_pgn = pgn + san + " "
        board.push(move)
        try:
            if board.is_game_over():
                return None
            try:
                raw = self.oracle.policy(board, child_pgn)
            except Exception:
                return None
            if not raw:
                return None
            legal_set = set(board.legal_moves)
            replies = sorted(((p, m) for m, p in raw.items() if m in legal_set),
                             key=lambda x: -x[0])[:3]
            if not replies:
                return None
            worst = INF
            for _, rm in replies:
                board.push(rm)
                self.nodes = 0
                if board.is_checkmate():
                    val = -MATE
                else:
                    val = self._tactical(board, -INF, INF, QDEPTH, 2)
                board.pop()
                if val < worst:
                    worst = val
            return worst
        finally:
            board.pop()


def create_player(oracle, budget):
    return Player(oracle, budget)
