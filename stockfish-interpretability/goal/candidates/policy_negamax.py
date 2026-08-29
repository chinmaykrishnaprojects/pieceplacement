"""policy_negamax - policy-pruned best-first negamax with a strong static leaf evaluation.

Idea
----
The language model is used ONLY as a move-ordering / pruning oracle: at any node
we ask it once for its human-move distribution and keep just the top few moves.
Everything else - the actual judgement of a position - is done by our own free
arithmetic: a tapered piece-square / pawn-structure / mobility / king-safety
evaluation, with a static-exchange-driven quiescence search over captures so we
never score a position in the middle of a trade.

Search shape: best-first negamax.  We hold a tiny explicit tree; every node's
value is either its quiescence score (leaf) or max(-child) (interior).  Each LM
call expands exactly one node, and we always spend the next call on the leaf that
currently sits on the principal variation.  That makes the budget adaptive on its
own: in a quiet position the first few expansions confirm the top move and the
tree stays flat, while in a sharp position the PV keeps flipping and the calls
get poured into the critical line.  The tree that results is small enough to read
off by hand - which is the whole point of the exercise.

Budget schedule (calls = expansions, never more than `budget`):
  budget 1  -> root expansion only: policy candidates scored by quiescence
               (a strict improvement on plain argmax, at the same cost)
  budget b  -> root breadth min(5, max(2, b)); reply breadth 2 (b<=3) else 3;
               b-1 further PV expansions, hard depth cap 6 plies.
"""

import chess

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

MATE = 30000
INF = 1 << 24

PV = (0, 100, 320, 330, 500, 900, 20000)   # indexed by chess piece type

QDEPTH = 5
QNODES = 500
MAXDEPTH = 6
PRIOR_W = 55.0          # centipawn weight on the LM prior at the root
UNVERIFIED = 25         # discount for a root move we never got to expand


def _rows(rows):
    """rows[0] is rank 8 ... rows[7] is rank 1 -> table indexed by square."""
    t = []
    for r in range(8):
        t.extend(rows[7 - r])
    return tuple(t)


PST_PAWN_MG = _rows([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [5, 5, 10, 25, 25, 10, 5, 5],
    [0, 0, 0, 20, 20, 0, 0, 0],
    [5, -5, -10, 0, 0, -10, -5, 5],
    [5, 10, 10, -20, -20, 10, 10, 5],
    [0, 0, 0, 0, 0, 0, 0, 0]])

PST_PAWN_EG = _rows([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [90, 90, 90, 90, 90, 90, 90, 90],
    [55, 55, 55, 55, 55, 55, 55, 55],
    [30, 30, 30, 30, 30, 30, 30, 30],
    [18, 18, 18, 18, 18, 18, 18, 18],
    [8, 8, 8, 8, 8, 8, 8, 8],
    [6, 6, 6, 6, 6, 6, 6, 6],
    [0, 0, 0, 0, 0, 0, 0, 0]])

PST_KNIGHT = _rows([
    [-50, -40, -30, -30, -30, -30, -40, -50],
    [-40, -20, 0, 0, 0, 0, -20, -40],
    [-30, 0, 10, 15, 15, 10, 0, -30],
    [-30, 5, 15, 20, 20, 15, 5, -30],
    [-30, 0, 15, 20, 20, 15, 0, -30],
    [-30, 5, 10, 15, 15, 10, 5, -30],
    [-40, -20, 0, 5, 5, 0, -20, -40],
    [-50, -40, -30, -30, -30, -30, -40, -50]])

PST_BISHOP = _rows([
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-10, 0, 5, 10, 10, 5, 0, -10],
    [-10, 5, 5, 10, 10, 5, 5, -10],
    [-10, 0, 10, 10, 10, 10, 0, -10],
    [-10, 10, 10, 10, 10, 10, 10, -10],
    [-10, 5, 0, 0, 0, 0, 5, -10],
    [-20, -10, -10, -10, -10, -10, -10, -20]])

PST_ROOK = _rows([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [5, 10, 10, 10, 10, 10, 10, 5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [0, 0, 0, 5, 5, 0, 0, 0]])

PST_QUEEN = _rows([
    [-20, -10, -10, -5, -5, -10, -10, -20],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-10, 0, 5, 5, 5, 5, 0, -10],
    [-5, 0, 5, 5, 5, 5, 0, -5],
    [0, 0, 5, 5, 5, 5, 0, -5],
    [-10, 5, 5, 5, 5, 5, 0, -10],
    [-10, 0, 5, 0, 0, 0, 0, -10],
    [-20, -10, -10, -5, -5, -10, -10, -20]])

PST_KING_MG = _rows([
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-20, -30, -30, -40, -40, -30, -30, -20],
    [-10, -20, -20, -20, -20, -20, -20, -10],
    [20, 20, 0, 0, 0, 0, 20, 20],
    [20, 30, 10, 0, 0, 10, 30, 20]])

PST_KING_EG = _rows([
    [-50, -40, -30, -20, -20, -30, -40, -50],
    [-30, -20, -10, 0, 0, -10, -20, -30],
    [-30, -10, 20, 30, 30, 20, -10, -30],
    [-30, -10, 30, 40, 40, 30, -10, -30],
    [-30, -10, 30, 40, 40, 30, -10, -30],
    [-30, -10, 20, 30, 30, 20, -10, -30],
    [-30, -30, 0, 0, 0, 0, -30, -30],
    [-50, -30, -30, -30, -30, -30, -30, -50]])

# mirrored copies for black (square_mirror flips the rank)
_MIR = tuple(chess.square_mirror(s) for s in range(64))


def _mirror(tbl):
    return tuple(tbl[_MIR[s]] for s in range(64))


PST = {
    chess.WHITE: {
        chess.KNIGHT: PST_KNIGHT, chess.BISHOP: PST_BISHOP,
        chess.ROOK: PST_ROOK, chess.QUEEN: PST_QUEEN},
    chess.BLACK: {
        chess.KNIGHT: _mirror(PST_KNIGHT), chess.BISHOP: _mirror(PST_BISHOP),
        chess.ROOK: _mirror(PST_ROOK), chess.QUEEN: _mirror(PST_QUEEN)},
}
PAWN_MG = {chess.WHITE: PST_PAWN_MG, chess.BLACK: _mirror(PST_PAWN_MG)}
PAWN_EG = {chess.WHITE: PST_PAWN_EG, chess.BLACK: _mirror(PST_PAWN_EG)}
KING_MG = {chess.WHITE: PST_KING_MG, chess.BLACK: _mirror(PST_KING_MG)}
KING_EG = {chess.WHITE: PST_KING_EG, chess.BLACK: _mirror(PST_KING_EG)}

FILE_BB = chess.BB_FILES
ADJ_FILES = tuple(
    (FILE_BB[f - 1] if f > 0 else 0) | (FILE_BB[f + 1] if f < 7 else 0)
    for f in range(8))

# squares strictly ahead of `sq` on its own and adjacent files
FRONT_SPAN = {chess.WHITE: [], chess.BLACK: []}
FRONT_FILE = {chess.WHITE: [], chess.BLACK: []}
for _sq in range(64):
    _f = _sq & 7
    _r = _sq >> 3
    _ahead_w = 0
    _ahead_b = 0
    for _rr in range(_r + 1, 8):
        _ahead_w |= chess.BB_RANKS[_rr]
    for _rr in range(0, _r):
        _ahead_b |= chess.BB_RANKS[_rr]
    FRONT_SPAN[chess.WHITE].append(_ahead_w & (FILE_BB[_f] | ADJ_FILES[_f]))
    FRONT_SPAN[chess.BLACK].append(_ahead_b & (FILE_BB[_f] | ADJ_FILES[_f]))
    FRONT_FILE[chess.WHITE].append(_ahead_w & FILE_BB[_f])
    FRONT_FILE[chess.BLACK].append(_ahead_b & FILE_BB[_f])

PASSED_BONUS = (0, 8, 14, 24, 44, 76, 120, 0)   # by relative rank of the pawn
SHIELD_MISSING = 16
KING_ATT_W = (0, 0, 12, 12, 20, 32, 0)          # per piece type attacking ring

_popcount = getattr(int, "bit_count", None)
if _popcount is None:                            # older interpreters
    def _pc(x):
        return bin(x).count("1")
else:
    def _pc(x):
        return x.bit_count()


def _bishop_att(sq, occ):
    return chess.BB_DIAG_ATTACKS[sq][occ & chess.BB_DIAG_MASKS[sq]]


def _rook_att(sq, occ):
    return (chess.BB_RANK_ATTACKS[sq][occ & chess.BB_RANK_MASKS[sq]] |
            chess.BB_FILE_ATTACKS[sq][occ & chess.BB_FILE_MASKS[sq]])


def _squares(bb):
    while bb:
        low = bb & -bb
        yield low.bit_length() - 1
        bb ^= low


# --------------------------------------------------------------------------
# static evaluation (free - no LM involved)
# --------------------------------------------------------------------------

def evaluate(board):
    """Centipawn score from the perspective of the side to move."""
    occ = board.occupied
    pawns = board.pawns
    knights = board.knights
    bishops = board.bishops
    rooks = board.rooks
    queens = board.queens

    co = board.occupied_co
    wp = co[chess.WHITE]
    bp = co[chess.BLACK]

    # ---- game phase (24 = full material, 0 = bare endgame)
    phase = (_pc((knights | bishops)) * 1 + _pc(rooks) * 2 + _pc(queens) * 4)
    if phase > 24:
        phase = 24
    mg_w = phase / 24.0
    eg_w = 1.0 - mg_w

    kings = (board.king(chess.WHITE), board.king(chess.BLACK))
    ring = [0, 0]
    for c in (chess.WHITE, chess.BLACK):
        ks = kings[c]
        ring[c] = (chess.BB_KING_ATTACKS[ks] | (1 << ks)) if ks is not None else 0

    katt = [0, 0]          # attack pressure ON the king of colour i
    score = 0.0            # white perspective

    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        mine = co[color]
        theirs = co[not color]
        enemy_ring = ring[not color]
        sub = 0.0

        my_pawns = pawns & mine
        their_pawns = pawns & theirs

        # ---- pawns: material, PST, doubled / isolated / passed
        pfiles = [0] * 8
        for sq in _squares(my_pawns):
            pfiles[sq & 7] += 1
        for sq in _squares(my_pawns):
            f = sq & 7
            sub += 100
            sub += PAWN_MG[color][sq] * mg_w + PAWN_EG[color][sq] * eg_w
            if pfiles[f] > 1:
                sub -= 11
            if not (my_pawns & ADJ_FILES[f]):
                sub -= 14
            if not (their_pawns & FRONT_SPAN[color][sq]) and \
                    not (my_pawns & FRONT_FILE[color][sq]):
                rr = (sq >> 3) if color == chess.WHITE else 7 - (sq >> 3)
                sub += PASSED_BONUS[rr] * (0.6 + 0.6 * eg_w)

        # ---- knights
        for sq in _squares(knights & mine):
            sub += 320 + PST[color][chess.KNIGHT][sq]
            a = chess.BB_KNIGHT_ATTACKS[sq] & ~mine
            sub += 4 * _pc(a)
            if a & enemy_ring:
                katt[not color] += KING_ATT_W[chess.KNIGHT]

        # ---- bishops
        bb_mine = bishops & mine
        if _pc(bb_mine) >= 2:
            sub += 32
        for sq in _squares(bb_mine):
            sub += 330 + PST[color][chess.BISHOP][sq]
            a = _bishop_att(sq, occ) & ~mine
            sub += 4 * _pc(a)
            if a & enemy_ring:
                katt[not color] += KING_ATT_W[chess.BISHOP]

        # ---- rooks
        for sq in _squares(rooks & mine):
            sub += 500 + PST[color][chess.ROOK][sq]
            a = _rook_att(sq, occ) & ~mine
            sub += 3 * _pc(a)
            if a & enemy_ring:
                katt[not color] += KING_ATT_W[chess.ROOK]
            fb = FILE_BB[sq & 7]
            if not (my_pawns & fb):
                sub += 18 if not (their_pawns & fb) else 9
            rr = (sq >> 3) if color == chess.WHITE else 7 - (sq >> 3)
            if rr == 6:
                sub += 18

        # ---- queens
        for sq in _squares(queens & mine):
            sub += 900 + PST[color][chess.QUEEN][sq]
            a = (_rook_att(sq, occ) | _bishop_att(sq, occ)) & ~mine
            sub += 1 * _pc(a)
            if a & enemy_ring:
                katt[not color] += KING_ATT_W[chess.QUEEN]

        # ---- king placement + pawn shield
        ks = kings[color]
        if ks is not None:
            sub += KING_MG[color][ks] * mg_w + KING_EG[color][ks] * eg_w
            f = ks & 7
            r = ks >> 3
            band = 0
            for dr in (1, 2):
                rr = r + dr if color == chess.WHITE else r - dr
                if 0 <= rr <= 7:
                    band |= chess.BB_RANKS[rr]
            missing = 0
            for df in (-1, 0, 1):
                nf = f + df
                if nf < 0 or nf > 7:
                    continue
                if not (my_pawns & FILE_BB[nf] & band):
                    missing += 1
            sub -= SHIELD_MISSING * missing * mg_w

        score += sign * sub

    # king attack pressure (scaled by phase - irrelevant in bare endings)
    score -= katt[chess.WHITE] * mg_w
    score += katt[chess.BLACK] * mg_w

    if board.turn == chess.WHITE:
        return score + 12
    return -score + 12


# --------------------------------------------------------------------------
# static exchange evaluation
# --------------------------------------------------------------------------

def _attackers(board, color, sq, occ):
    them = board.occupied_co[color] & occ
    res = chess.BB_PAWN_ATTACKS[not color][sq] & board.pawns & them
    res |= chess.BB_KNIGHT_ATTACKS[sq] & board.knights & them
    res |= chess.BB_KING_ATTACKS[sq] & board.kings & them
    res |= _bishop_att(sq, occ) & (board.bishops | board.queens) & them
    res |= _rook_att(sq, occ) & (board.rooks | board.queens) & them
    return res


_ORDER = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK,
          chess.QUEEN, chess.KING)


def _least_valuable(board, atts):
    for pt in _ORDER:
        bb = atts & board.pieces_mask(pt, chess.WHITE) | \
            atts & board.pieces_mask(pt, chess.BLACK)
        if bb:
            low = bb & -bb
            return low.bit_length() - 1, pt
    return None, None


def see(board, move):
    """Static exchange evaluation of `move` (centipawns, mover's view)."""
    to_sq = move.to_square
    occ = board.occupied
    if board.is_en_passant(move):
        gain0 = 100
        cap_sq = to_sq + (-8 if board.turn == chess.WHITE else 8)
        occ &= ~(1 << cap_sq)
        occ |= (1 << to_sq)
    else:
        victim = board.piece_type_at(to_sq)
        gain0 = PV[victim] if victim else 0
    apt = board.piece_type_at(move.from_square)
    if apt is None:
        return 0
    if move.promotion:
        gain0 += PV[move.promotion] - 100
        apt = move.promotion
    asq = move.from_square
    side = not board.turn

    gain = [0] * 34
    gain[0] = gain0
    d = 0
    while True:
        d += 1
        if d > 31:
            break
        gain[d] = PV[apt] - gain[d - 1]
        if max(-gain[d - 1], gain[d]) < 0:
            break
        occ &= ~(1 << asq)
        atts = _attackers(board, side, to_sq, occ)
        if not atts:
            break
        asq, apt = _least_valuable(board, atts)
        if asq is None:
            break
        side = not side
    d -= 1
    while d > 0:
        a = -gain[d - 1]
        b = gain[d]
        gain[d - 1] = -(a if a > b else b)
        d -= 1
    return gain[0]


# --------------------------------------------------------------------------
# quiescence search (free - own evaluation only)
# --------------------------------------------------------------------------

def _qsearch(board, alpha, beta, depth, ctr):
    ctr[0] += 1
    if ctr[0] > QNODES:
        return evaluate(board)

    if board.is_check():
        moves = list(board.legal_moves)
        if not moves:
            return -MATE + 100
        if depth <= 0:
            return evaluate(board)
        best = -INF
        moves.sort(key=lambda m: _cap_key(board, m), reverse=True)
    else:
        stand = evaluate(board)
        if stand >= beta:
            return stand
        if stand > alpha:
            alpha = stand
        if depth <= 0:
            return stand
        best = stand
        moves = []
        for m in board.generate_legal_captures():
            v = board.piece_type_at(m.to_square)
            gain = PV[v] if v else 100
            if m.promotion:
                gain += PV[m.promotion] - 100
            if stand + gain + 180 < alpha:
                continue
            if see(board, m) < 0:
                continue
            moves.append(m)
        moves.sort(key=lambda m: _cap_key(board, m), reverse=True)

    for m in moves:
        board.push(m)
        v = -_qsearch(board, -beta, -alpha, depth - 1, ctr)
        board.pop()
        if v > best:
            best = v
            if best > alpha:
                alpha = best
                if alpha >= beta:
                    break
    return best


def _cap_key(board, m):
    v = board.piece_type_at(m.to_square)
    a = board.piece_type_at(m.from_square)
    return (PV[v] if v else 0) * 16 - (PV[a] if a else 0) + \
        (900 if m.promotion else 0)


def quiesce(board):
    return _qsearch(board, -INF, INF, QDEPTH, [0])


# --------------------------------------------------------------------------
# search tree
# --------------------------------------------------------------------------

class Node:
    __slots__ = ("board", "pgn", "parent", "move", "prior", "san",
                 "children", "val", "depth", "terminal", "expanded")

    def __init__(self, board, pgn, parent, move, prior, depth, san=""):
        self.board = board
        self.pgn = pgn
        self.parent = parent
        self.move = move
        self.prior = prior
        self.san = san
        self.children = []
        self.depth = depth
        self.expanded = False
        self.terminal = False
        self.val = 0.0


def _has_mate_in_1(board):
    """True if the side to move can mate immediately (free - no LM call)."""
    for m in board.legal_moves:
        if not board.gives_check(m):
            continue
        board.push(m)
        done = board.is_checkmate()
        board.pop()
        if done:
            return True
    return False


def _terminal_value(board, depth):
    """Decisive score if the position is over, else None."""
    if board.is_checkmate():
        return -(MATE - depth * 10)
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    if board.halfmove_clock >= 100:
        return 0.0
    if board.halfmove_clock >= 8 and board.is_repetition(3):
        return 0.0
    return None


class Player:
    def __init__(self, oracle, budget):
        self.oracle = oracle
        self.budget = int(budget)
        self.used = 0
        self.lines = []          # human-legible record of what was considered

    # -- LM access -------------------------------------------------------
    def _policy(self, node):
        if self.used >= self.budget:
            return None
        self.used += 1
        try:
            return self.oracle.policy(node.board, node.pgn)
        except Exception:            # BudgetExhausted or model hiccup
            self.used = self.budget
            return None

    # -- tree ------------------------------------------------------------
    def _leaf_value(self, board, depth):
        t = _terminal_value(board, depth)
        if t is not None:
            return t, True
        return float(quiesce(board)), False

    def _make_child(self, node, move, prior):
        b = node.board
        san = b.san(move)
        if b.turn == chess.WHITE:
            pgn = node.pgn + "%d.%s " % (b.fullmove_number, san)
        else:
            pgn = node.pgn + "%s " % san
        nb = b.copy(stack=12)
        nb.push(move)
        ch = Node(nb, pgn, node, move, prior, node.depth + 1, san)
        v, term = self._leaf_value(nb, ch.depth)
        if not term and ch.depth == 1 and _has_mate_in_1(nb):
            # the reply mates us: quiescence would never notice a quiet mate
            v = MATE - 20
            term = True
        ch.val = v
        ch.terminal = term
        return ch

    def _expand(self, node, breadth, extras=False):
        pol = self._policy(node)
        node.expanded = True
        cands = []
        if pol:
            legal = set(node.board.legal_moves)
            items = [(m, p) for m, p in pol.items() if m in legal]
            items.sort(key=lambda kv: kv[1], reverse=True)
            total = sum(p for _, p in items) or 1.0
            mass = 0.0
            for m, p in items:
                cands.append((m, p / total))
                mass += p / total
                if len(cands) >= breadth or (len(cands) >= 2 and mass >= 0.95):
                    break
        if extras:
            have = {m for m, _ in cands}
            tact = []
            for m in node.board.generate_legal_captures():
                if m in have:
                    continue
                s = see(node.board, m)
                if s > 0:
                    tact.append((s, m))
            tact.sort(key=lambda t: t[0], reverse=True)
            for _, m in tact[:2]:
                cands.append((m, 0.0))
        if not cands:
            return
        for m, p in cands:
            node.children.append(self._make_child(node, m, p))
        self._backup(node)

    def _backup(self, node):
        while node is not None:
            if node.children:
                node.val = max(-c.val for c in node.children)
            node = node.parent

    def _pick_leaf(self, node):
        """First expandable node in principal-variation order."""
        if not node.children:
            if node.expanded or node.terminal or node.depth >= MAXDEPTH:
                return None
            return node
        order = sorted(node.children, key=lambda c: -c.val, reverse=True)
        for c in order:
            got = self._pick_leaf(c)
            if got is not None:
                return got
        return None

    # -- top level -------------------------------------------------------
    def _search(self, board, pgn):
        root = Node(board.copy(stack=12), pgn, None, None, 1.0, 0)
        b = self.budget

        if b <= 1:
            root_k, reply_k = 5, 2
        else:
            root_k = min(5, max(2, b))
            reply_k = 2 if b <= 3 else 3

        self._expand(root, root_k, extras=True)
        if not root.children:
            return None

        while self.used < self.budget:
            leaf = self._pick_leaf(root)
            if leaf is None:
                break
            before = self.used
            # augment shallow reply sets with free material grabs the LM's
            # human-move distribution may simply not mention
            self._expand(leaf, reply_k, extras=(leaf.depth <= 2))
            if self.used == before:      # no call happened -> budget gone
                break

        prior_w = 70.0 if b <= 1 else PRIOR_W
        best = None
        best_score = -INF
        for c in root.children:
            s = -c.val + prior_w * c.prior
            if not c.expanded and not c.terminal:
                s -= UNVERIFIED
            if s > best_score:
                best_score = s
                best = c
        self.lines = [(c.san, round(-c.val, 1), round(c.prior, 3),
                       len(c.children)) for c in root.children]
        return best.move if best is not None else None

    def play(self, board, pgn):
        self.used = 0          # the harness re-arms the quota every move
        legal = list(board.legal_moves)
        if not legal:
            return None
        # free tactical shortcut: an immediate mate needs no model call
        for m in legal:
            board.push(m)
            done = board.is_checkmate()
            board.pop()
            if done:
                return m
        fallback = legal[0]
        try:
            mv = self._search(board, pgn)
        except Exception:                # never forfeit on our own bugs
            mv = None
        if mv is not None and mv in board.legal_moves:
            return mv
        # last resort without any further model access: pure static choice
        try:
            best, bv = fallback, -INF
            for m in legal:
                board.push(m)
                v = -float(evaluate(board))
                board.pop()
                if v > bv:
                    bv, best = v, m
            return best
        except Exception:
            return fallback


def create_player(oracle, budget):
    return Player(oracle, budget)
